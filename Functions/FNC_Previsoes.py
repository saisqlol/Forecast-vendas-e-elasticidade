import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import os

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
    df_previsao_completo = pd.read_excel(caminho_planilha_previsao)
    
    # Garantir que a coluna SKU seja do mesmo tipo (string) para a comparação
    df_previsao_completo['SKU'] = df_previsao_completo['SKU'].astype(str)
    sku = str(sku) # Garantir que o SKU de entrada também seja string
    
    # Filtrar o DataFrame de previsão para conter apenas o SKU em análise
    df_previsao = df_previsao_completo[df_previsao_completo['SKU'] == sku].copy()

    if df_previsao.empty:
        print(f"AVISO: Nenhum dado encontrado para o SKU {sku} no arquivo de previsão. A função será encerrada.")
        # Retorna DataFrames vazios para evitar erros no notebook
        return pd.DataFrame(), pd.DataFrame()

    df_previsao['Data'] = pd.to_datetime(df_previsao['Data'])
    df_previsao['Log_Preco'] = np.log(df_previsao['Preco'].clip(lower=0.01))

    # -- Previsão com Modelo TSCV (Validação Cruzada) --
    print(f"\nCalculando previsões para o SKU {sku} com o modelo de Validação Cruzada (TSCV)...")
    
    # Criar features dummies necessárias
    df_previsao['Quarta-feira'] = (df_previsao['Data'].dt.dayofweek == 2).astype(int)
    df_previsao['Terça-feira'] = (df_previsao['Data'].dt.dayofweek == 1).astype(int)
    
    for col in X_cols_tscv:
        if col not in df_previsao.columns:
            raise ValueError(f"Coluna necessária para o modelo TSCV não encontrada: {col}")
            
    X_tscv = df_previsao[X_cols_tscv]
    
    log_demanda_tscv = resultados_tscv['intercepto'] + X_tscv.dot(resultados_tscv['coeficientes'])
    df_previsao['previsao_TSCV'] = np.exp(log_demanda_tscv)
    
    # -- Previsão com Modelo SARIMAX (Iterativa para Robustez) --
    print(f"Calculando previsões para o SKU {sku} com o modelo SARIMAX...")
    
    exog_cols_sarimax = [col for col in ['Log_Preco', 'Quarta-feira', 'Terça-feira'] if col in df_previsao.columns]
    df_previsao_indexed = df_previsao.set_index('Data')
    
    previsoes_sarimax_log = []
    for data_atual in df_previsao_indexed.index:
        exog_atual = df_previsao_indexed.loc[[data_atual], exog_cols_sarimax]
        # Garantir que o índice tenha a frequência correta ('D' para diário)
        exog_atual.index.freq = 'D'
        
        previsao_passo = resultados_sarimax.predict(
            start=data_atual,
            end=data_atual,
            exog=exog_atual
        )
        previsoes_sarimax_log.append(previsao_passo.iloc[0])

    df_previsao['previsao_SARIMAX_log'] = previsoes_sarimax_log
    df_previsao['previsao_SARIMAX'] = np.exp(df_previsao['previsao_SARIMAX_log'])
    
    # --- Lógica para a coluna 'previsao_total' ---
    # 1. Determinar o modelo ideal com base no AIC e RMSE
    aic_tscv = resultados_tscv['metricas_medias'].get('aic', np.inf)
    aic_sarimax = resultados_sarimax.aic
    
    rmse_tscv = resultados_tscv['metricas_medias'].get('rmse', np.inf)
    rmse_sarimax = getattr(resultados_sarimax, 'custom_metrics', {}).get('rmse', np.inf)

    # Regra de decisão:
    # 1. O modelo com menor AIC é o preferido.
    # 2. REGRA DE DESEMPATE/SEGURANÇA: Se o erro (RMSE) do TSCV for maior que o do SARIMAX,
    #    o SARIMAX é escolhido, mesmo que seu AIC seja um pouco maior.
    modelo_ideal_aic = 'TSCV' if aic_tscv < aic_sarimax else 'SARIMAX'
    
    if rmse_tscv > rmse_sarimax:
        modelo_ideal = 'SARIMAX'
        print(f"  DECISÃO: SARIMAX escolhido como modelo ideal (RMSE TSCV {rmse_tscv:.4f} > RMSE SARIMAX {rmse_sarimax:.4f})")
    else:
        modelo_ideal = modelo_ideal_aic
        print(f"  DECISÃO: {modelo_ideal} escolhido como modelo ideal (baseado no AIC)")

    # 2. Criar a previsão total com base no modelo ideal
    df_previsao['previsao_total'] = np.where(
        modelo_ideal == 'TSCV',
        df_previsao['previsao_TSCV'],
        df_previsao['previsao_SARIMAX']
    )
    
    # 3. Aplicar a regra de segurança: se SARIMAX for >50% menor que TSCV, somar os dois
    condicao_override = df_previsao['previsao_SARIMAX'] < (0.5 * df_previsao['previsao_TSCV'])
    df_previsao['previsao_total'] = np.where(
        condicao_override,
        df_previsao['previsao_SARIMAX'] + df_previsao['previsao_TSCV'],
        df_previsao['previsao_total']
    )

    # Consolidar resultado das previsões
    df_resultado_previsoes = df_previsao[['Data', 'SKU', 'Preco', 'previsao_SARIMAX', 'previsao_TSCV', 'previsao_total']].copy()
    
    caminho_csv_previsoes = f'../Resultados/previsoes_consolidadas_{sku}.csv'
    df_resultado_previsoes.to_csv(caminho_csv_previsoes, index=False, sep=';', decimal=',')
    print(f"\nArquivo de previsões salvo em: {caminho_csv_previsoes}")
    
    # Gerar o relatório de comparação, agora passando o modelo_ideal
    df_relatorio_modelos = gerar_relatorio_comparacao(
        resultados_tscv, 
        resultados_sarimax, 
        sku, 
        X_cols_tscv
    )
    
    caminho_csv_relatorio = f'../Resultados/relatorio_comparacao_modelos_{sku}.csv'
    df_relatorio_modelos.to_csv(caminho_csv_relatorio, index=False, sep=';', decimal=',')
    print(f"Arquivo de relatório de modelos salvo em: {caminho_csv_relatorio}")
    
    print("\n--- Processo Concluído ---")
    
    return df_resultado_previsoes, df_relatorio_modelos

def gerar_relatorio_comparacao(resultados_tscv, resultados_sarimax, sku, X_cols_tscv):
    """
    Gera um DataFrame com o relatório de comparação dos resultados dos modelos.

    Args:
        resultados_tscv (dict): Dicionário com os resultados do modelo de validação cruzada.
        resultados_sarimax (SARIMAXResults): Objeto com os resultados do modelo SARIMAX.
        sku (str): O SKU do produto.
        X_cols_tscv (list): Lista de colunas de features usadas no modelo TSCV.

    Returns:
        pd.DataFrame: DataFrame com o relatório consolidado para o SKU.
    """
    # --- Lógica para determinar o modelo ideal ---
    aic_tscv = resultados_tscv['metricas_medias'].get('aic', np.inf)
    aic_sarimax = resultados_sarimax.aic
    
    rmse_tscv = resultados_tscv['metricas_medias'].get('rmse', np.inf)
    rmse_sarimax = getattr(resultados_sarimax, 'custom_metrics', {}).get('rmse', np.inf)

    modelo_ideal_aic = 'TSCV' if aic_tscv < aic_sarimax else 'ARIMAX'
    
    if rmse_tscv > rmse_sarimax:
        modelo_ideal = 'ARIMAX'
    else:
        modelo_ideal = modelo_ideal_aic

    # Extrair coeficientes do TSCV de forma segura
    coeficientes_tscv = {}
    for col in ['Log_Preco', 'Quarta-feira', 'Terça-feira']:
        try:
            idx = X_cols_tscv.index(col)
            coeficientes_tscv[f'coef_{col.lower()}_tscv'] = resultados_tscv['coeficientes'][idx]
        except (ValueError, IndexError):
            coeficientes_tscv[f'coef_{col.lower()}_tscv'] = np.nan

    dados_relatorio = {
        'sku': sku,
        'data_rodagem': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'modelo_ideal': modelo_ideal,
        'intercepto_tscv': resultados_tscv.get('intercepto', np.nan),
        **coeficientes_tscv, # Adiciona os coeficientes do TSCV
        'intercepto_sarimax': resultados_sarimax.params.get('intercept', np.nan),
        'coef_log_preco_sarimax': resultados_sarimax.params.get('Log_Preco', np.nan),
        'coef_quarta-feira_sarimax': resultados_sarimax.params.get('Quarta-feira', np.nan),
        'coef_terça-feira_sarimax': resultados_sarimax.params.get('Terça-feira', np.nan),
        'AIC_sarimax': resultados_sarimax.aic,
        'BIC_sarimax': resultados_sarimax.bic,
        'AIC_cruzado': resultados_tscv['metricas_medias'].get('aic', np.nan),
        'BIC_cruzado': resultados_tscv['metricas_medias'].get('bic', np.nan)
    }
    
    return pd.DataFrame([dados_relatorio])

def pred_prox_30_dias(
    resultados_tscv,
    resultados_sarimax,
    df_venda,
    sku,
    X_cols_tscv=['Log_Preco', 'Quarta-feira', 'Terça-feira'],
    gerar_grafico=True
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
        gerar_grafico (bool): Se True, gera e salva um gráfico da previsão.

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

    # Previsão SARIMAX (usando .get_forecast() para projeção futura contínua)
    exog_cols_sarimax = [col for col in ['Log_Preco', 'Quarta-feira', 'Terça-feira'] if col in df_futuro.columns]
    exog_futuro = df_futuro.set_index('Data')[exog_cols_sarimax]
    # Garantir que o índice tenha a frequência correta ('D' para diário)
    exog_futuro.index.freq = 'D'

    # .get_forecast() é o método mais robusto para previsões dinâmicas out-of-sample.
    # Ele projeta o futuro passo a passo, usando a previsão anterior como input para a próxima.
    forecast_results = resultados_sarimax.get_forecast(steps=len(df_futuro), exog=exog_futuro)
    log_demanda_sarimax = forecast_results.predicted_mean
    df_futuro['previsao_SARIMAX'] = np.exp(log_demanda_sarimax.values)

    # --- Lógica para a coluna 'previsao_total' ---
    # 1. Determinar o modelo ideal com base no AIC e RMSE
    aic_tscv = resultados_tscv['metricas_medias'].get('aic', np.inf)
    aic_sarimax = resultados_sarimax.aic
    
    rmse_tscv = resultados_tscv['metricas_medias'].get('rmse', np.inf)
    rmse_sarimax = getattr(resultados_sarimax, 'custom_metrics', {}).get('rmse', np.inf)

    modelo_ideal_aic = 'TSCV' if aic_tscv < aic_sarimax else 'SARIMAX'
    
    if rmse_tscv > rmse_sarimax:
        modelo_ideal = 'SARIMAX'
    else:
        modelo_ideal = modelo_ideal_aic

    # 2. Criar a previsão total com base no modelo ideal
    df_futuro['previsao_total'] = np.where(
        modelo_ideal == 'TSCV',
        df_futuro['previsao_TSCV'],
        df_futuro['previsao_SARIMAX']
    )
    
    # 3. Aplicar a regra de segurança
    condicao_override = df_futuro['previsao_SARIMAX'] < (0.5 * df_futuro['previsao_TSCV'])
    df_futuro['previsao_total'] = np.where(
        condicao_override,
        df_futuro['previsao_SARIMAX'] + df_futuro['previsao_TSCV'],
        df_futuro['previsao_total']
    )

    if gerar_grafico:
        print("  Gerando gráfico de previsão futura...")
        # 3. Preparar dados para o gráfico
        df_historico = df_venda.last('30D')
        df_futuro_plot = df_futuro.set_index('Data')

        # 4. Gerar o gráfico
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(18, 8))

        plt.plot(df_historico.index, df_historico['Demanda'], label='Demanda Real (Últimos 30 dias)', color='black', marker='o', markersize=4, linestyle='--')
        plt.plot(df_futuro_plot.index, df_futuro_plot['previsao_SARIMAX'], label='Previsão SARIMAX', color='red', linewidth=1.5, alpha=0.7)
        plt.plot(df_futuro_plot.index, df_futuro_plot['previsao_TSCV'], label='Previsão TSCV', color='blue', linewidth=1.5, alpha=0.7)
        plt.plot(df_futuro_plot.index, df_futuro_plot['previsao_total'], label='Previsão Ideal', color='green', linewidth=2.5)

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

def prever_demanda_com_modelos_salvos(caminho_pasta_modelos, caminho_planilha_precos):
    """
    Gera previsões de demanda utilizando os arquivos de modelo salvos.

    Args:
        caminho_pasta_modelos (str): Caminho para a pasta onde os modelos (.joblib e .pkl) estão salvos.
        caminho_planilha_precos (str): Caminho para a planilha Excel com os SKUs e preços para prever.

    Returns:
        pd.DataFrame: DataFrame com as previsões para os SKUs encontrados em ambos os locais.
    """
    print("--- INICIANDO PREVISÃO A PARTIR DE MODELOS SALVOS ---")

    # 1. Ler a planilha de preços
    try:
        df_precos = pd.read_excel(caminho_planilha_precos)
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo de preços não encontrado - {e}")
        return pd.DataFrame()

    df_precos['SKU'] = df_precos['SKU'].astype(str)
    skus_para_prever = df_precos['SKU'].unique()
    print(f"Encontrados {len(skus_para_prever)} SKUs no arquivo de preços.")

    # 2. Iterar e gerar previsões
    previsoes_finais = []
    for sku in skus_para_prever:
        print(f"  Processando SKU: {sku}")
        dados_sku_precos = df_precos[df_precos['SKU'] == sku].copy()
        
        # Caminhos para os arquivos de modelo
        caminho_modelo_tscv = os.path.join(caminho_pasta_modelos, f'modelo_tscv_{sku}.joblib')
        caminho_modelo_sarimax = os.path.join(caminho_pasta_modelos, f'modelo_sarimax_{sku}.pkl')

        if not os.path.exists(caminho_modelo_tscv) or not os.path.exists(caminho_modelo_sarimax):
            print(f"    AVISO: Arquivos de modelo para o SKU {sku} não encontrados. Pulando.")
            continue

        # Carregar modelos
        modelo_tscv_carregado = joblib.load(caminho_modelo_tscv)
        modelo_sarimax_carregado = SARIMAXResults.load(caminho_modelo_sarimax)

        # Preparar features
        dados_sku_precos['Data'] = pd.to_datetime(dados_sku_precos['Data'])
        dados_sku_precos['Log_Preco'] = np.log(dados_sku_precos['Preco'].clip(lower=0.01))
        dados_sku_precos['Quarta-feira'] = (dados_sku_precos['Data'].dt.dayofweek == 2).astype(int)
        dados_sku_precos['Terça-feira'] = (dados_sku_precos['Data'].dt.dayofweek == 1).astype(int)

        # Previsão TSCV (pode ser feita em lote)
        X_cols_tscv = ['Log_Preco', 'Quarta-feira', 'Terça-feira']
        X_tscv = dados_sku_precos[X_cols_tscv]
        log_demanda_tscv = modelo_tscv_carregado['intercepto'] + X_tscv.dot(modelo_tscv_carregado['coeficientes'])
        dados_sku_precos['previsao_TSCV'] = np.exp(log_demanda_tscv)

        # Previsão SARIMAX (deve ser feita de forma iterativa para modelos carregados)
        previsoes_sarimax_log = []
        df_previsao_indexed = dados_sku_precos.set_index('Data')
        exog_cols_sarimax = [col for col in ['Log_Preco', 'Quarta-feira', 'Terça-feira'] if col in df_previsao_indexed.columns]

        for data_atual in df_previsao_indexed.index:
            exog_atual = df_previsao_indexed.loc[[data_atual], exog_cols_sarimax]
            # Garantir que o índice tenha a frequência correta ('D' para diário)
            exog_atual.index.freq = 'D'
            
            # Faz a previsão para um único passo
            previsao_passo = modelo_sarimax_carregado.predict(
                start=data_atual,
                end=data_atual,
                exog=exog_atual
            )
            previsoes_sarimax_log.append(previsao_passo.iloc[0])

        dados_sku_precos['previsao_SARIMAX_log'] = previsoes_sarimax_log
        dados_sku_precos['previsao_SARIMAX'] = np.exp(dados_sku_precos['previsao_SARIMAX_log'])
        
        previsoes_finais.append(dados_sku_precos)

    if not previsoes_finais:
        print("Nenhuma previsão pôde ser gerada.")
        return pd.DataFrame()

    # 4. Consolidar e retornar
    df_final = pd.concat(previsoes_finais, ignore_index=True)
    colunas_resultado = ['Data', 'SKU', 'Preco', 'previsao_TSCV', 'previsao_SARIMAX']
    
    print("\n--- PREVISÃO CONCLUÍDA ---")
    return df_final[colunas_resultado]
