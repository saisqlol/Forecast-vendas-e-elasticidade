import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
import pmdarima as pm

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


def modelo_sarimax(df, sku, *exog_vars, endog_var='Log_Demanda', order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), trend=None, verbose=True):
    """
    Cria e treina um modelo SARIMAX para prever a demanda.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados de vendas.
        sku (str): O SKU do produto a ser modelado.
        *exog_vars: Nomes das colunas a serem usadas como variáveis exógenas.
        endog_var (str): Nome da coluna a ser usada como variável endógena (dependente).
        order (tuple): A ordem (p, d, q) do modelo.
        seasonal_order (tuple): A ordem sazonal (P, D, Q, s) do modelo.
        trend (str, optional): Parâmetro de tendência para o modelo ('c' para constante/intercepto).
        verbose (bool): Se True, imprime logs detalhados.

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
    
    # Para a variável endógena (Log_Demanda), preencher os dias sem venda com o valor logarítmico
    # que representa a demanda zero (np.log(0.01)), em vez de 0.
    valor_demanda_zero = np.log(0.01)
    df_resampled[endog_var] = df_resampled[endog_var].fillna(valor_demanda_zero) 

    # 1. Verificação de estacionariedade
    if verbose:
        # Opcional: preencher a demanda original com 0 para a verificação de estacionariedade
        if 'Demanda' in df_resampled.columns:
            df_resampled['Demanda'] = df_resampled['Demanda'].fillna(0)
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
    
    if verbose:
        print(f"\n--- Treinando Modelo SARIMAX para SKU: {sku} ---")
        print(f"Variável Dependente: {endog_var}")
        print(f"Variáveis Independentes: {exog_list}")
        print(f"Ordem (p,d,q): {order}")
        print(f"Ordem Sazonal (P,D,Q,s): {seasonal_order}")
        print(f"Trend: {trend}\n")

    # 3. Criação e treinamento do modelo SARIMAX
    try:
        modelo = sm.tsa.statespace.SARIMAX(
            endog,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        resultado = modelo.fit(disp=False)
        
        if verbose:
            print("--- Sumário do Modelo SARIMAX ---")
            print(resultado.summary())
        
        # Cálculo de métricas de erro (RMSE e WAPE)
        y_true = endog
        y_pred = resultado.fittedvalues
        
        # Alinhar as séries e remover NaNs para cálculo
        df_comp = pd.DataFrame({'true': y_true, 'pred': y_pred}).dropna()
        
        if not df_comp.empty:
            # RMSE (Root Mean Squared Error)
            rmse = np.sqrt(np.mean((df_comp['true'] - df_comp['pred'])**2))
            
            # WAPE (Weighted Absolute Percentage Error)
            # Evitar divisão por zero se a soma dos valores verdadeiros for 0
            sum_true = np.sum(np.abs(df_comp['true']))
            if sum_true > 0:
                wape = np.sum(np.abs(df_comp['true'] - df_comp['pred'])) / sum_true * 100
            else:
                wape = np.nan # Não é possível calcular se a soma for zero
            
            # Anexar métricas ao objeto de resultado para uso posterior no pipeline
            resultado.custom_metrics = {'rmse': rmse, 'wape': wape}

            if verbose:
                print(f"\nMétricas de Ajuste do Modelo (In-sample):")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  WAPE: {wape:.2f}%")
        elif verbose:
            print("\nNão foi possível calcular métricas de ajuste (sem dados sobrepostos).")

        if verbose:
            print("-" * 50)
        
        return resultado

    except Exception as e:
        if verbose:
            print(f"Ocorreu um erro ao treinar o modelo SARIMAX para o SKU {sku}: {e}")
        return None

def encontrar_melhores_parametros_sarimax(df, sku, exog_vars, endog_var='Log_Demanda', verbose=True):
    """
    Usa auto_arima para encontrar os melhores parâmetros (p,d,q) para um modelo ARIMAX.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados de vendas.
        sku (str): O SKU do produto a ser modelado.
        exog_vars (list): Lista de nomes de colunas a serem usadas como variáveis exógenas.
        endog_var (str): Nome da coluna a ser usada como variável endógena.
        verbose (bool): Se True, imprime logs detalhados.

    Returns:
        tuple: Contendo (order, seasonal_order, trend) ótimos para o modelo.
    """
    if verbose:
        print(f"\n--- Buscando Melhores Parâmetros ARIMAX para SKU: {sku} ---")
    
    # Preparar os dados da mesma forma que no modelo principal
    df_resampled = df.asfreq('D')
    for col in exog_vars:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].fillna(method='ffill').fillna(method='bfill')
    
    valor_demanda_zero = np.log(0.01)
    df_resampled[endog_var] = df_resampled[endog_var].fillna(valor_demanda_zero)

    endog = df_resampled[endog_var]
    exog_list = [var for var in exog_vars if var in df_resampled.columns]
    exog = df_resampled[exog_list] if exog_list else None

    # Usar auto_arima para encontrar os melhores parâmetros
    try:
        if verbose:
            print("\n--- Iniciando busca de parâmetros para modelo de Regressão com Erros ARMA (d=0, D=0) ---")
        # Forçamos d=0 e D=0. Isso cria um modelo de regressão com erros ARMA, que é muito mais estável
        # para previsão quando se tem variáveis exógenas fortes.
        # As variáveis exógenas (preço, dias da semana) definem o nível da previsão,
        # enquanto os componentes ARMA/SARMA modelam a estrutura de correlação dos resíduos.
        auto_model = pm.auto_arima(
            y=endog,
            X=exog,
            start_p=1, start_q=1,
            test='adf',       
            max_p=3, max_q=3,
            m=1,              # Desativar a sazonalidade, pois já temos dummies de dia da semana
            d=0,              # Forçar não-diferenciação para estabilidade
            seasonal=False,   # Desativar completamente a busca por parâmetros sazonais (P, D, Q)
            start_P=0, 
            D=0,              
            trace=verbose,    # Controla o log do auto_arima
            error_action='ignore',  
            suppress_warnings=True, 
            stepwise=True,
            n_jobs=-1
        )

        if verbose:
            print(auto_model.summary())
        
        # Extrair os melhores parâmetros encontrados
        best_order = auto_model.order
        best_seasonal_order = auto_model.seasonal_order
        
        # Forçar o modelo a sempre ter um intercepto ('c' de constante) para garantir estabilidade na previsão.
        # A busca automática (auto_arima) pode às vezes remover o intercepto para otimizar o AIC,
        # mas isso pode levar a previsões futuras instáveis e irrealistas.
        trend_term = 'c'
        
        if verbose:
            print(f"Melhores parâmetros encontrados: order={best_order}, seasonal_order={best_seasonal_order}, trend='{trend_term}' (forçado)")
        return best_order, best_seasonal_order, trend_term

    except Exception as e:
        if verbose:
            print(f"Ocorreu um erro ao buscar os parâmetros para o SKU {sku}: {e}")
            print("Retornando para os parâmetros padrão (1,1,1)(0,0,0,0) e com trend 'c'.")
        return (1, 1, 1), (0, 0, 0, 0), 'c'

