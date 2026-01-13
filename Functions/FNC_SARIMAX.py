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
    if series_sem_na.empty:
        print(f"Série '{nome_serie}' está vazia após remover NAs. Não é possível verificar estacionariedade.")
        return

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
        print("Nenhuma das variáveis exógenas especificadas foi encontrada no DataFrame. Treinando modelo sem variáveis exógenas.")
        exog = None
    else:
        exog = df_resampled[exog_list]
    
    if verbose:
        print(f"\n--- Treinando Modelo SARIMAX para SKU: {sku} ---")
        print(f"Variável Dependente: {endog_var}")
        if exog is not None:
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
    Usa auto_arima para encontrar os melhores parâmetros (p,d,q)(P,D,Q,s) para um modelo SARIMAX.
    Esta função é otimizada para encontrar um modelo de regressão com erros SARIMA,
    que é mais estável para previsão quando se tem variáveis exógenas fortes.
    """
    if verbose:
        print(f"\n--- Buscando Melhores Parâmetros SARIMAX para SKU: {sku} ---")
    
    # Preparar os dados da mesma forma que no modelo principal
    df_resampled = df.asfreq('D')
    for col in exog_vars:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].fillna(method='ffill').fillna(method='bfill')
    
    valor_demanda_zero = np.log(0.01)
    df_resampled[endog_var] = df_resampled[endog_var].fillna(valor_demanda_zero)

    endog = df_resampled[endog_var].dropna()
    exog_list = [var for var in exog_vars if var in df_resampled.columns]
    
    if not exog_list:
        exog = None
    else:
        # Alinhar exog com endog
        exog = df_resampled.loc[endog.index, exog_list]

    try:
        if verbose:
            print("\n--- Iniciando busca de parâmetros com auto_arima ---")
        
        # auto_arima vai testar diferentes combinações de p, d, q, P, D, Q
        auto_model = pm.auto_arima(
            y=endog,
            X=exog,
            start_p=1, start_q=1,
            test='adf',       
            max_p=3, max_q=3,
            m=7,              # Sazonalidade semanal
            d=None,           # Deixar o auto_arima encontrar a melhor ordem de diferenciação
            seasonal=True,    # Ativar a busca por parâmetros sazonais
            start_P=1, start_Q=1,
            max_P=2, max_Q=2,
            D=None,           # Deixar o auto_arima encontrar a melhor ordem de diferenciação sazonal
            trace=verbose,
            error_action='ignore',  
            suppress_warnings=True, 
            stepwise=True,
            n_jobs=-1
        )

        if verbose:
            print(auto_model.summary())
        
        best_order = auto_model.order
        best_seasonal_order = auto_model.seasonal_order
        
        # O auto_arima pode incluir um termo de tendência/intercepto automaticamente.
        # Vamos extrair se ele foi adicionado para passar para o nosso modelo final.
        trend_term = None
        if auto_model.with_intercept():
            trend_term = 'c'
        
        if verbose:
            print(f"Melhores parâmetros encontrados: order={best_order}, seasonal_order={best_seasonal_order}, trend='{trend_term}'")
        return best_order, best_seasonal_order, trend_term

    except Exception as e:
        if verbose:
            print(f"Ocorreu um erro ao buscar os parâmetros para o SKU {sku}: {e}")
            print("Retornando para os parâmetros padrão (1,1,1)(1,1,1,7) e com trend 'c'.")
        return (1, 1, 1), (1, 1, 1, 7), 'c'


def gerar_previsoes_vmd(resultado_modelo, df_original, sku, exog_vars, window_days=60, verbose=False, clip_factor=3):
    """
    Calcula o VMD realizado e prevê o VMD para o dia seguinte e para a próxima semana.

    Args:
        resultado_modelo (statsmodels.results.sarimax.SARIMAXResults): O modelo SARIMAX treinado.
        df_original (pd.DataFrame): O DataFrame original com os dados históricos do SKU.
        sku (str): O SKU do produto.
        exog_vars (list): Lista com o nome das variáveis exógenas utilizadas no modelo.
        window_days (int): Número de dias para calcular o VMD (padrão 60).
        verbose (bool): Se True, imprime informações de debug sobre as previsões.
        clip_factor (float): Fator multiplicador para limitar previsões acima do histórico (padrão 3).

    Returns:
        dict: Um dicionário com o VMD realizado e os VMDs previstos.
    """
    if resultado_modelo is None or df_original.empty:
        return {
            'SKU': sku,
            f'VMD_Realizado_{window_days}D': np.nan,
            'VMD_Previsto_Amanha': np.nan,
            'VMD_Previsto_7D': np.nan,
            'Status': 'Erro: Modelo ou dados de entrada inválidos.'
        }

    # --- 1. Calcular VMD Realizado (últimos window_days dias) ---
    hoje = df_original.index.max()
    inicio_periodo = hoje - pd.Timedelta(days=window_days - 1)
    demanda_ultimos_window = df_original.loc[inicio_periodo:hoje, 'Demanda']

    # Se não houver dados suficientes, usar o que estiver disponível
    if demanda_ultimos_window.empty:
        return {
            'SKU': sku,
            f'VMD_Realizado_{window_days}D': np.nan,
            'VMD_Previsto_Amanha': np.nan,
            'VMD_Previsto_7D': np.nan,
            'Status': 'Erro: Sem dados de demanda para o período solicitado.'
        }
    
    vmd_realizado = demanda_ultimos_window.mean()

    # --- 2. Preparar DataFrame para Previsão (próximos 7 dias) ---
    ultima_data = df_original.index.max()
    datas_futuras = pd.date_range(start=ultima_data + pd.Timedelta(days=1), periods=7, freq='D')
    df_futuro = pd.DataFrame(index=datas_futuras)

    # Preencher variáveis exógenas para o futuro
    # Preço: Assume que o preço do último dia se mantém
    df_futuro['Preco'] = df_original['Preco'].iloc[-1]
    df_futuro['Log_Preco'] = np.log(df_futuro['Preco'].clip(lower=0.01))

    # Dias da semana
    df_futuro['Dia_Semana'] = df_futuro.index.day_name(locale='pt_BR')
    dias_da_semana_dummies = pd.get_dummies(df_futuro['Dia_Semana'], dtype=int)
    # Garante que todas as colunas de dias da semana possíveis existam
    dias_poss_veis = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
    for dia in dias_poss_veis:
        if dia not in dias_da_semana_dummies.columns:
            dias_da_semana_dummies[dia] = 0
    df_futuro = pd.concat([df_futuro, dias_da_semana_dummies], axis=1)

    # Black Friday e Promoções: Assumir como 0 (não ocorrendo)
    df_futuro['Black_Friday'] = ((df_futuro.index.month == 11) & (df_futuro.index.day >= 28) & (df_futuro.index.day <= 30)).astype(int)
    df_futuro['promocionado'] = 0 # Assumindo sem promoção

    # Garantir que todas as colunas exógenas do modelo existam no df_futuro
    for col in exog_vars:
        if col not in df_futuro.columns:
            df_futuro[col] = 0 

    exog_futuro = df_futuro[exog_vars]

    # --- 3. Gerar Previsões de Demanda ---
    previsoes_log = resultado_modelo.get_forecast(steps=7, exog=exog_futuro)
    previsoes_demanda = np.exp(previsoes_log.predicted_mean).round() # Arredonda para o inteiro mais próximo

    # Garantir que a previsão não seja negativa
    previsoes_demanda[previsoes_demanda < 0] = 0

    # --- 4. Calcular VMDs Previstos ---
    n = len(demanda_ultimos_window)
    denom = window_days if n >= window_days else n

    # VMD para Amanhã (últimos window_days-1 dias reais + 1 dia previsto)
    if n >= 2:
        soma_ultimos_minus1 = demanda_ultimos_window.iloc[1:].sum()
    else:
        soma_ultimos_minus1 = 0
    vmd_previsto_amanha = (soma_ultimos_minus1 + previsoes_demanda.iloc[0]) / denom

    # VMD para a Próxima Semana (últimos window_days-7 dias reais + 7 dias previstos)
    if n > 7:
        soma_ultimos_minus7 = demanda_ultimos_window.iloc[7:].sum()
    else:
        soma_ultimos_minus7 = 0
    vmd_previsto_7d = (soma_ultimos_minus7 + previsoes_demanda.sum()) / denom

    return {
        'SKU': sku,
        f'VMD_Realizado_{window_days}D': vmd_realizado,
        'VMD_Previsto_Amanha': vmd_previsto_amanha,
        'VMD_Previsto_7D': vmd_previsto_7d,
        'Status': 'Sucesso'
    }
