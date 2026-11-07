from pathlib import Path

import pandas as pd
import streamlit as st

CAMINHO_MODELO = Path("modelo/modelo_previnad.joblib")

st.set_page_config(page_title="PrevInad", layout="centered")


@st.cache_resource
def carregar_modelo():
    """Carrega o pipeline treinado. Devolve None enquanto o arquivo não existe."""
    if CAMINHO_MODELO.exists():
        import joblib
        return joblib.load(CAMINHO_MODELO)
    return None


def prever_probabilidade(dados: pd.DataFrame, modelo) -> float:
    """Probabilidade de inadimplência entre 0 e 1."""
    if modelo is not None:
        return float(modelo.predict_proba(dados)[0][1])

    # Cálculo provisório só para testar a tela. Apagar quando o modelo real chegar.
    linha = dados.iloc[0]
    pontos = (
        0.45 * linha["uso_limite_credito"]
        + 0.25 * min(linha["razao_endividamento"], 1.0)
        + 0.08 * min(linha["qtd_atrasos_30_59_dias"], 5)
        - 0.002 * (linha["idade"] - 18)
    )
    return max(0.01, min(0.99, pontos))


def faixa_de_risco(prob: float) -> str:
    if prob < 0.20:
        return "Risco baixo"
    if prob < 0.50:
        return "Risco moderado"
    return "Risco alto"


modelo = carregar_modelo()

st.title("PrevInad")
st.caption("Alerta precoce de inadimplência. Protótipo da disciplina SIN 323, UFV.")

if modelo is None:
    st.warning("Modelo treinado ainda não encontrado. A tela usa um cálculo provisório.")

st.sidebar.header("Perfil do cliente")
idade = st.sidebar.slider("Idade", 18, 90, 40)
renda_mensal = st.sidebar.number_input("Renda mensal (R$)", 0, 100000, 4000, step=500)
uso_limite_credito = st.sidebar.slider("Uso do limite de crédito", 0.0, 1.5, 0.3, step=0.05)
razao_endividamento = st.sidebar.slider("Razão de endividamento", 0.0, 2.0, 0.4, step=0.05)
qtd_atrasos_30_59_dias = st.sidebar.slider("Atrasos de 30 a 59 dias", 0, 10, 0)

dados = pd.DataFrame(
    [
        {
            "uso_limite_credito": uso_limite_credito,
            "idade": idade,
            "qtd_atrasos_30_59_dias": qtd_atrasos_30_59_dias,
            "razao_endividamento": razao_endividamento,
            "renda_mensal": renda_mensal,
        }
    ]
)

if st.button("Calcular risco", type="primary"):
    prob = prever_probabilidade(dados, modelo)

    st.subheader(faixa_de_risco(prob))
    st.metric("Probabilidade de inadimplência", f"{prob:.1%}")
    st.progress(prob)

    with st.expander("Dados enviados ao modelo"):
        st.dataframe(dados)
