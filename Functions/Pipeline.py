import pandas as pd
import traceback
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Importar todas as funções necessárias para o pipeline
from Functions.FNC_Pro import Base_venda
from Functions.FNC_TSCV import modelo_validacao_cruzada_series_temporais
from Functions.FNC_SARIMAX import modelo_sarimax
from Functions.FNC_Previsoes import gerar_relatorio_comparacao, pred_prox_30_dias

def pipeline_completa_skus(df_produtos, caminho_planilha_precos, n_splits=10):
    """
    Pipeline completa para processar uma lista de SKUs, treinando modelos,
    gerando relatórios de comparação e previsões futuras.

    Args:
        df_produtos (pd.DataFrame): DataFrame com a coluna 'ID_Sku'.
        caminho_planilha_precos (str): Caminho para a planilha Excel com preços para previsão.
        n_splits (int): Número de folds para a validação cruzada.

    Returns:
        tuple: Contendo (df_relatorios_final, df_previsoes_futuras_final).
    """
    
    print("--- INICIANDO PIPELINE COMPLETA PARA TODOS OS SKUs ---")
    print(f"Total de SKUs para processar: {len(df_produtos)}")
    print("-" * 60)
    
    # Listas para consolidar os resultados de todos os SKUs
    relatorios_consolidados = []
    previsoes_futuras_consolidadas = []
    
    skus_processados = 0
    skus_com_erro = 0
    
    for idx, row in df_produtos.iterrows():
        sku = str(row['ID_Sku']).strip()
        
        print(f"\n>>> Processando SKU {sku} ({idx + 1}/{len(df_produtos)}) <<<")
        
        try:
            # 1. Obter base de vendas
            Venda = Base_venda(sku)
            
            if Venda is None or len(Venda) < 30:
                print(f"  AVISO: SKU {sku} com dados insuficientes ({len(Venda) if Venda is not None else 0} registros). Pulando...")
                skus_com_erro += 1
                continue
            
            # 2. Treinar Modelo de Validação Cruzada (TSCV)
            print(f"  [1/4] Treinando modelo de Validação Cruzada...")
            X_cols_tscv = ['Log_Preco', 'Quarta-feira', 'Terça-feira']
            resultados_tscv = modelo_validacao_cruzada_series_temporais(
                Venda, sku, *X_cols_tscv, var_dpd='Log_Demanda', n_splits=n_splits
            )

            # 3. Treinar Modelo SARIMAX
            print(f"  [2/4] Treinando modelo SARIMAX...")
            exog_vars_sarimax = ['Log_Preco']
            resultado_sarimax = modelo_sarimax(
                Venda, sku, *exog_vars_sarimax, endog_var='Log_Demanda',
                order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)
            )

            # 4. Gerar Relatório de Comparação de Modelos
            print(f"  [3/4] Gerando relatório de comparação de modelos...")
            df_relatorio = gerar_relatorio_comparacao(
                resultados_tscv, resultado_sarimax, sku, X_cols_tscv
            )
            relatorios_consolidados.append(df_relatorio)

            # 5. Gerar Previsão para os Próximos 30 Dias, sem gerar o gráfico
            print(f"  [4/4] Gerando previsão para os próximos 30 dias...")
            df_previsao_futura = pred_prox_30_dias(
                resultados_tscv, 
                resultado_sarimax, 
                Venda, 
                sku, 
                X_cols_tscv=X_cols_tscv,
                gerar_grafico=False  # Desativa a criação do gráfico no pipeline
            )
            previsoes_futuras_consolidadas.append(df_previsao_futura)
            
            skus_processados += 1
            print(f"  >>> SKU {sku} processado com sucesso! <<<")
            
        except Exception as e:
            print(f"  !!! ERRO no SKU {sku}: {str(e)} !!!")
            print(traceback.format_exc())
            skus_com_erro += 1
            
            # Adicionar um registro de erro ao relatório
            relatorios_consolidados.append(pd.DataFrame([{
                'sku': sku, 'data_rodagem': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Status': f'Erro: {str(e)}'
            }]))
            continue
    
    # --- Consolidação Final ---
    print("\n--- CONSOLIDANDO RESULTADOS FINAIS ---")
    
    # Consolidar e salvar relatórios
    df_relatorios_final = pd.concat(relatorios_consolidados, ignore_index=True)
    caminho_relatorio = f'../Resultados/Relatorio_Modelos_Consolidado_{datetime.now():%Y%m%d}.csv'
    df_relatorios_final.to_csv(caminho_relatorio, index=False, sep=';', decimal=',')
    print(f"Relatório de modelos consolidado salvo em: {caminho_relatorio}")

    # Consolidar e salvar previsões futuras
    if previsoes_futuras_consolidadas:
        df_previsoes_futuras_final = pd.concat(previsoes_futuras_consolidadas, ignore_index=True)
        caminho_previsoes = f'../Resultados/Previsoes_Futuras_Consolidadas_{datetime.now():%Y%m%d}.csv'
        df_previsoes_futuras_final.to_csv(caminho_previsoes, index=False, sep=';', decimal=',')
        print(f"Previsões futuras consolidadas salvas em: {caminho_previsoes}")
    else:
        df_previsoes_futuras_final = pd.DataFrame()

    print("-" * 60)
    print("--- PIPELINE CONCLUÍDA ---")
    print(f"  SKUs processados com sucesso: {skus_processados}")
    print(f"  SKUs com erro: {skus_com_erro}")
    
    return df_relatorios_final, df_previsoes_futuras_final