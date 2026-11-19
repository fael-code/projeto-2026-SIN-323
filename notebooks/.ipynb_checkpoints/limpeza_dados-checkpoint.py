import os
import pandas as pd

# 1. Carregamento dos dados brutos
caminho_bruto = "dados/brutos/cs-training.csv"
df_bruto = pd.read_csv(caminho_bruto)

# 2. Remoção da coluna de índice desnecessária (evita data leakage)
if "Unnamed: 0" in df_bruto.columns:
    df_bruto = df_bruto.drop(columns=["Unnamed: 0"])

# 3. Dicionário de mapeamento para padronização
mapa_colunas = {
    "SeriousDlqin2yrs": "inadimplente_target",
    "RevolvingUtilizationOfUnsecuredLines": "uso_limite_credito",
    "age": "idade",
    "NumberOfTime30-59DaysPastDueNotWorse": "qtd_atrasos_30_59_dias",
    "DebtRatio": "razao_endividamento",
    "MonthlyIncome": "renda_mensal",
    "NumberOfOpenCreditLinesAndLoans": "qtd_emprestimos_abertos",
    "NumberOfTimes90DaysLate": "qtd_atrasos_90_dias",
    "NumberRealEstateLoansOrLines": "qtd_emprestimos_imobiliarios",
    "NumberOfTime60-89DaysPastDueNotWorse": "qtd_atrasos_60_89_dias",
    "NumberOfDependents": "qtd_dependentes",
}
df_limpo = df_bruto.rename(columns=mapa_colunas)

# 4. Tratamento de valores ausentes (Imputação pela Mediana - mais robusta a outliers)
df_limpo["renda_mensal"] = df_limpo["renda_mensal"].fillna(
    df_limpo["renda_mensal"].median()
)
df_limpo["qtd_dependentes"] = df_limpo["qtd_dependentes"].fillna(0)

# 5. Filtragem básica de inconsistências / outliers severos
# Remove idades inválidas (<= 18 ou > 100) e códigos de erro nos atrasos (96 e 98)
df_limpo = df_limpo[(df_limpo["idade"] >= 18) & (df_limpo["idade"] <= 100)]
df_limpo = df_limpo[df_limpo["qtd_atrasos_30_59_dias"] < 90]
df_limpo = df_limpo[df_limpo["qtd_atrasos_60_89_dias"] < 90]
df_limpo = df_limpo[df_limpo["qtd_atrasos_90_dias"] < 90]

# 6. Exportação da base tratada
os.makedirs("dados/processados", exist_ok=True)
caminho_saida = "dados/processados/dados_inadimplencia_limpos.csv"
df_limpo.to_csv(caminho_saida, index=False)

print(f"Limpeza concluída com sucesso! Total de registros válidos: {len(df_limpo)}")
print("\nResumo de valores nulos restantes:")
print(df_limpo.isnull().sum())