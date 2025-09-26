import pandas as pd
import numpy as np
from datetime import datetime

def gerar_previsoes_e_relatorios(
    resultados_tscv, 
    resultados_sarimax, 
    sku, 
    caminho_planilha_previsao,
    X_cols_tscv=['Log_Preco', 'Quarta-feira', 'Terça-feira']
):
    """
    Gera previsões a partir dos modelos TSCV e SARIMAX e cria um relatório de comparação.

    Args:
        resultados_tscv (dict): Dicionário com os resultados do modelo de validação cruzada.
        resultados_sarimax (SARIMAXResults): Objeto com os resultados do modelo SARIMAX.
        sku (str): O SKU do produto.
        caminho_planilha_previsao (str): Caminho para a planilha Excel com os dados para previsão.
        X_cols_tscv (list): Lista de colunas de features usadas no modelo TSCV.

    Returns:
        tuple: Contendo (df_resultado_previsoes, df_relatorio_modelos).
    """
    print("--- INICIANDO GERAÇÃO DE PREVISÕES E RELATÓRIOS ---")
    
    # --- 1. GERAÇÃO DAS PREVISÕES ---
    
    # Ler e preparar dados de entrada
    df_previsao = pd.read_excel(caminho_planilha_previsao)
    df_previsao['Data'] = pd.to_datetime(df_previsao['Data'])
    df_previsao['Log_Preco'] = np.log(df_previsao['Preco'].clip(lower=0.01))

    # -- Previsão com Modelo TSCV (Validação Cruzada) --
    print("\nCalculando previsões para o modelo de Validação Cruzada (TSCV)...")
    
    # Criar features dummies necessárias
    df_previsao['Quarta-feira'] = (df_previsao['Data'].dt.dayofweek == 2).astype(int)
    df_previsao['Terça-feira'] = (df_previsao['Data'].dt.dayofweek == 1).astype(int)
    
    for col in X_cols_tscv:
        if col not in df_previsao.columns:
            raise ValueError(f"Coluna necessária para o modelo TSCV não encontrada: {col}")
            
    X_tscv = df_previsao[X_cols_tscv]
    
    log_demanda_tscv = resultados_tscv['intercepto'] + X_tscv.dot(resultados_tscv['coeficientes'])
    df_previsao['previsao_TSCV'] = np.exp(log_demanda_tscv)
    
    # -- Previsão com Modelo SARIMAX --
    print("Calculando previsões para o modelo SARIMAX...")
    
    exog_sarimax = df_previsao[['Log_Preco']]
    exog_sarimax.index = df_previsao['Data']
    
    log_demanda_sarimax = resultados_sarimax.forecast(steps=len(df_previsao), exog=exog_sarimax)
    df_previsao['previsao_SARIMAX'] = np.exp(log_demanda_sarimax.values)
    
    # Consolidar resultado das previsões
    df_resultado_previsoes = df_previsao[['Data', 'SKU', 'Preco', 'previsao_SARIMAX', 'previsao_TSCV']].copy()
    
    caminho_csv_previsoes = f'../Resultados/previsoes_consolidadas_{sku}.csv'
    df_resultado_previsoes.to_csv(caminho_csv_previsoes, index=False, sep=';', decimal=',')
    print(f"\nArquivo de previsões salvo em: {caminho_csv_previsoes}")

    # --- 2. GERAÇÃO DO RELATÓRIO DE COMPARAÇÃO DE MODELOS ---
    print("\nGerando relatório de comparação de modelos...")
    
    try:
        idx_log_preco_tscv = X_cols_tscv.index('Log_Preco')
        coef_log_preco_tscv = resultados_tscv['coeficientes'][idx_log_preco_tscv]
    except (ValueError, IndexError):
        coef_log_preco_tscv = np.nan

    coef_log_preco_sarimax = resultados_sarimax.params.get('Log_Preco', np.nan)

    dados_relatorio = {
        'sku': [sku],
        'data_rodagem': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'coef_log_preco_tscv': [coef_log_preco_tscv],
        'coef_log_preco_sarimax': [coef_log_preco_sarimax],
        'intercepto_tscv': [resultados_tscv.get('intercepto', np.nan)],
        'AIC_sarimax': [resultados_sarimax.aic],
        'BIC_sarimax': [resultados_sarimax.bic],
        'AIC_cruzado': [resultados_tscv['metricas_medias'].get('aic', np.nan)],
        'BIC_cruzado': [resultados_tscv['metricas_medias'].get('bic', np.nan)]
    }
    
    df_relatorio_modelos = pd.DataFrame(dados_relatorio)
    
    caminho_csv_relatorio = f'../Resultados/relatorio_comparacao_modelos_{sku}.csv'
    df_relatorio_modelos.to_csv(caminho_csv_relatorio, index=False, sep=';', decimal=',')
    print(f"Arquivo de relatório de modelos salvo em: {caminho_csv_relatorio}")
    
    print("\n--- Processo Concluído ---")
    
    return df_resultado_previsoes, df_relatorio_modelos
