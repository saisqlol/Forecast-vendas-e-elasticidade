import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

def verificar_estacionariedade(series, nome_serie=''):
    """
    Verifica a estacionariedade de uma série temporal usando o teste Augmented Dickey-Fuller (ADF).
    """
    print(f"--- Análise de Estacionariedade para: {nome_serie} ---")
    # O teste ADF requer que não haja valores ausentes
    series_sem_na = series.dropna()
    resultado_adf = adfuller(series_sem_na)
    
    print(f'Estatística ADF: {resultado_adf[0]}')
    print(f'p-valor: {resultado_adf[1]}')
    print('Valores Críticos:')
    for key, value in resultado_adf[4].items():
        print(f'\t{key}: {value}')
        
    if resultado_adf[1] <= 0.05:
        print(f"Resultado: A série '{nome_serie}' é ESTACIONÁRIA (p-valor <= 0.05). Rejeita-se a hipótese nula.")
    else:
        print(f"Resultado: A série '{nome_serie}' NÃO é ESTACIONÁRIA (p-valor > 0.05). Não se pode rejeitar a hipótese nula.")
    print("-" * 50)


def modelo_sarimax(df, sku, *exog_vars, endog_var='Log_Demanda', order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """
    Cria e treina um modelo SARIMAX para prever a demanda.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados de vendas.
        sku (str): O SKU do produto a ser modelado.
        *exog_vars: Nomes das colunas a serem usadas como variáveis exógenas.
        endog_var (str): Nome da coluna a ser usada como variável endógena (dependente).
        order (tuple): A ordem (p, d, q) do modelo.
        seasonal_order (tuple): A ordem sazonal (P, D, Q, s) do modelo.

    Returns:
        statsmodels.results.sarimax.SARIMAXResults: O resultado do modelo treinado.
    """
    if df.empty:
        print(f"DataFrame para o SKU {sku} está vazio. O modelo não pode ser treinado.")
        return None

    # Reamostrar para frequência diária para garantir a continuidade da série temporal
    df_resampled = df.asfreq('D')

    # Preencher valores ausentes que podem ter surgido após a reamostragem
    # Para as variáveis exógenas, o preenchimento para frente é uma abordagem razoável
    for col in exog_vars:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].fillna(method='ffill').fillna(method='bfill')
    
    # Para a variável endógena, a interpolação pode ser uma opção, ou preenchimento com 0 se fizer sentido
    df_resampled[endog_var] = df_resampled[endog_var].fillna(0) 

    # 1. Verificação de estacionariedade
    if 'Demanda' in df_resampled.columns:
        verificar_estacionariedade(df_resampled['Demanda'], nome_serie='Demanda')
    if endog_var in df_resampled.columns:
        verificar_estacionariedade(df_resampled[endog_var], nome_serie=endog_var)

    # 2. Preparação das variáveis
    endog = df_resampled[endog_var]
    
    # Garantir que todas as variáveis exógenas existam no DataFrame
    exog_list = [var for var in exog_vars if var in df_resampled.columns]
    if not exog_list:
        print("Nenhuma das variáveis exógenas especificadas foi encontrada no DataFrame.")
        return None
        
    exog = df_resampled[exog_list]
    
    print(f"\n--- Treinando Modelo SARIMAX para SKU: {sku} ---")
    print(f"Variável Dependente: {endog_var}")
    print(f"Variáveis Independentes: {exog_list}")
    print(f"Ordem (p,d,q): {order}")
    print(f"Ordem Sazonal (P,D,Q,s): {seasonal_order}\n")

    # 3. Criação e treinamento do modelo SARIMAX
    try:
        modelo = sm.tsa.statespace.SARIMAX(
            endog,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        resultado = modelo.fit(disp=False)
        
        print("--- Sumário do Modelo SARIMAX ---")
        print(resultado.summary())
        print("-" * 50)
        
        return resultado

    except Exception as e:
        print(f"Ocorreu um erro ao treinar o modelo SARIMAX para o SKU {sku}: {e}")
        return None

