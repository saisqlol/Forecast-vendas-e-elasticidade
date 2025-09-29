import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

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

def pred_prox_30_dias(
    resultados_tscv,
    resultados_sarimax,
    df_venda,
    sku,
    X_cols_tscv=['Log_Preco', 'Quarta-feira', 'Terça-feira']
):
    """
    Prevê a demanda para os próximos 30 dias com base no último preço conhecido
    e gera um gráfico comparativo.

    Args:
        resultados_tscv (dict): Resultados do modelo de validação cruzada.
        resultados_sarimax (SARIMAXResults): Resultados do modelo SARIMAX.
        df_venda (pd.DataFrame): DataFrame com os dados históricos de vendas.
        sku (str): O SKU do produto.
        X_cols_tscv (list): Lista de colunas de features usadas no modelo TSCV.

    Returns:
        pd.DataFrame: DataFrame com as previsões para os próximos 30 dias.
    """
    print(f"--- INICIANDO PREVISÃO PARA OS PRÓXIMOS 30 DIAS (SKU: {sku}) ---")

    # 1. Preparar dados para a previsão futura
    if df_venda.empty:
        print("DataFrame de vendas está vazio. Não é possível fazer a previsão.")
        return None

    ultimo_preco = df_venda['Preco'].iloc[-1]
    ultima_data = df_venda.index.max()

    print(f"Último preço registrado: {ultimo_preco:.2f} em {ultima_data.date()}")
    print("Para os próximos 30 dias, o preço será baseado no mesmo dia 30 dias atrás, ou no último preço se a data não existir.")

    datas_futuras = pd.date_range(start=ultima_data + pd.Timedelta(days=1), periods=30, freq='D')
    
    # Lógica para determinar os preços futuros
    precos_futuros = []
    for data_futura in datas_futuras:
        data_passada = data_futura - pd.Timedelta(days=30)
        if data_passada in df_venda.index:
            preco_usado = df_venda.loc[data_passada, 'Preco']
        else:
            preco_usado = ultimo_preco  # Fallback
        precos_futuros.append(preco_usado)

    df_futuro = pd.DataFrame({
        'Data': datas_futuras,
        'SKU': sku,
        'Preco': precos_futuros
    })

    # 2. Gerar previsões para os modelos
    df_futuro['Log_Preco'] = np.log(df_futuro['Preco'].clip(lower=0.01))
    df_futuro['Quarta-feira'] = (df_futuro['Data'].dt.dayofweek == 2).astype(int)
    df_futuro['Terça-feira'] = (df_futuro['Data'].dt.dayofweek == 1).astype(int)

    # Previsão TSCV
    X_tscv = df_futuro[X_cols_tscv]
    log_demanda_tscv = resultados_tscv['intercepto'] + X_tscv.dot(resultados_tscv['coeficientes'])
    df_futuro['previsao_TSCV'] = np.exp(log_demanda_tscv)

    # Previsão SARIMAX
    exog_sarimax = df_futuro[['Log_Preco']]
    exog_sarimax.index = df_futuro['Data']
    log_demanda_sarimax = resultados_sarimax.forecast(steps=30, exog=exog_sarimax)
    df_futuro['previsao_SARIMAX'] = np.exp(log_demanda_sarimax.values)

    # 3. Preparar dados para o gráfico
    df_historico = df_venda.last('30D')
    df_futuro_plot = df_futuro.set_index('Data')

    # 4. Gerar o gráfico
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(18, 8))

    plt.plot(df_historico.index, df_historico['Demanda'], label='Demanda Real (Últimos 30 dias)', color='black', marker='o', markersize=4, linestyle='--')
    plt.plot(df_futuro_plot.index, df_futuro_plot['previsao_SARIMAX'], label='Previsão SARIMAX (Próximos 30 dias)', color='red', linewidth=2)
    plt.plot(df_futuro_plot.index, df_futuro_plot['previsao_TSCV'], label='Previsão TSCV (Próximos 30 dias)', color='blue', linewidth=2)

    plt.title(f'Previsão de Demanda para os Próximos 30 Dias - SKU {sku}')
    plt.xlabel('Data')
    plt.ylabel('Demanda')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adicionar uma linha vertical para separar o histórico da previsão
    plt.axvline(ultima_data, color='gray', linestyle=':', linewidth=2)

    caminho_grafico = f'../Graficos/previsao_30_dias_sku_{sku}.png'
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    print(f"\nGráfico de previsão salvo em: {caminho_grafico}")
    plt.show()

    return df_futuro
