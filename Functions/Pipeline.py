import pandas as pd
import traceback
from datetime import datetime
import warnings
import joblib
import os
import numpy as np

warnings.filterwarnings('ignore')

# Importar todas as funções necessárias para o pipeline
from Functions.FNC_Pro import Base_venda
from Functions.FNC_TSCV import modelo_validacao_cruzada_series_temporais
from Functions.FNC_ARIMAX import modelo_sarimax, encontrar_melhores_parametros_sarimax
from Functions.FNC_Previsoes import gerar_relatorio_comparacao

def pipeline_completa_skus(df_produtos, n_splits=10):
    """
    Pipeline completa para processar uma lista de SKUs, treinando modelos
    e gerando um relatório de comparação consolidado.

    Args:
        df_produtos (pd.DataFrame): DataFrame com a coluna 'ID_Sku'.
        n_splits (int): Número de folds para a validação cruzada.

    Returns:
        pd.DataFrame: DataFrame com o relatório consolidado dos modelos.
    """
    
    print("--- INICIANDO PIPELINE DE TREINAMENTO DE MODELOS ---")
    print(f"Total de SKUs para processar: {len(df_produtos)}")
    print("-" * 60)
    
    relatorios_consolidados = []
    skus_processados = 0
    skus_com_erro = 0
    
    for i, row in enumerate(df_produtos.itertuples(), 1):
        sku = str(row.ID_Sku).strip()
        
        print(f"\n>>> Processando SKU {sku} ({i}/{len(df_produtos)}) <<<")
        
        try:
            Venda = Base_venda(sku)
            
            if Venda is None or len(Venda) < 30:
                print(f"  AVISO: Dados insuficientes ({len(Venda) if Venda is not None else 0} registros). Pulando...")
                skus_com_erro += 1
                continue
            
            # --- Modelo TSCV ---
            X_cols_tscv = ['Log_Preco', 'Quarta-feira', 'Terça-feira', 'promocionado']
            resultados_tscv = modelo_validacao_cruzada_series_temporais(
                Venda, sku, *X_cols_tscv, var_dpd='Log_Demanda', n_splits=n_splits, verbose=False
            )
            print("  Modelo TSCV:")
            print(f"    AIC: {resultados_tscv['metricas_medias'].get('aic', 'N/A'):.2f}, BIC: {resultados_tscv['metricas_medias'].get('bic', 'N/A'):.2f}")
            print(f"    Coeficientes: {resultados_tscv['coeficientes']}")

            # --- Modelo ARIMAX ---
            exog_vars_arimax = ['Log_Preco', 'Quarta-feira', 'Terça-feira', 'promocionado']
            best_order, best_seasonal_order, best_trend = encontrar_melhores_parametros_sarimax(
                Venda, sku, exog_vars=exog_vars_arimax, endog_var='Log_Demanda', verbose=False
            )
            resultado_sarimax = modelo_sarimax(
                Venda, sku, *exog_vars_arimax, endog_var='Log_Demanda',
                order=best_order, seasonal_order=best_seasonal_order, trend=best_trend, verbose=False
            )
            print("  Modelo ARIMAX:")
            print(f"    AIC: {resultado_sarimax.aic:.2f}, BIC: {resultado_sarimax.bic:.2f}")
            print(f"    Coeficientes: {resultado_sarimax.params.to_dict()}")

            # --- Salvar Modelos ---
            caminho_modelo_tscv = f'../Modelos/modelo_tscv_{sku}.joblib'
            caminho_modelo_sarimax = f'../Modelos/modelo_sarimax_{sku}.pkl'
            joblib.dump(resultados_tscv, caminho_modelo_tscv)
            resultado_sarimax.save(caminho_modelo_sarimax)

            # --- Gerar e Consolidar Relatório ---
            df_relatorio = gerar_relatorio_comparacao(
                resultados_tscv, resultado_sarimax, sku, X_cols_tscv
            )
            relatorios_consolidados.append(df_relatorio)
            
            skus_processados += 1
            
        except Exception as e:
            print(f"  !!! ERRO no SKU {sku}: {str(e)} !!!")
            print(traceback.format_exc())
            skus_com_erro += 1
            relatorios_consolidados.append(pd.DataFrame([{'sku': sku, 'Status': f'Erro: {str(e)}'}]))
            continue
    
    # --- Consolidação Final ---
    print("\n--- CONSOLIDANDO RESULTADOS FINAIS ---")
    df_relatorios_final = pd.concat(relatorios_consolidados, ignore_index=True)
    caminho_relatorio = f'../Resultados/Relatorio_Modelos_Consolidado_{datetime.now():%Y%m%d}.csv'
    df_relatorios_final.to_csv(caminho_relatorio, index=False, sep=';', decimal=',')
    print(f"Relatório de modelos consolidado salvo em: {caminho_relatorio}")

    print("-" * 60)
    print("--- PIPELINE DE TREINAMENTO CONCLUÍDA ---")
    print(f"  SKUs processados com sucesso: {skus_processados}")
    print(f"  SKUs com erro: {skus_com_erro}")
    
    return df_relatorios_final