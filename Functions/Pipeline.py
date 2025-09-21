import pandas as pd
import numpy as np
import traceback
from datetime import datetime
import warnings
from google.cloud import bigquery
import os
warnings.filterwarnings('ignore')

# Importar suas funções existentes
from Functions.FNC_Pro import Base_venda
from Functions.FNC_TSCV import modelo_validacao_cruzada_series_temporais
from Functions.FNC_Previsoes import prever_planilha


# Puxar autorização para rodar a base de venda

def configurar_credenciais_bq(arquivo_json):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = arquivo_json

def pipeline_completa_skus(df_produtos, caminho_planilha_precos, n_splits=10):
    """
    Pipeline completa para processar todos os SKUs
    
    Parameters:
    df_produtos: DataFrame com coluna 'ID_Sku' contendo os SKUs
    caminho_planilha_precos: caminho para a planilha de preços para previsão
    n_splits: número de folds para validação cruzada
    
    Returns:
    DataFrame consolidado com resultados de todos os SKUs
    """
    
    print(" INICIANDO PIPELINE COMPLETA PARA TODOS OS SKUs")
    print("=" * 60)
    print(f"SKUs encontrados: {len(df_produtos)}")
    print(f"SKUs: {df_produtos['ID_Sku'].tolist()}")
    print("=" * 60)
    
    # DataFrame para consolidar resultados
    resultados_consolidados = []
    previsoes_consolidadas = []
    
    skus_processados = 0
    skus_com_erro = 0
    
    for idx, row in df_produtos.iterrows():
        sku = str(row['ID_Sku']).strip()
        
        print(f"\n Processando SKU {sku} ({idx + 1}/{len(df_produtos)})...")
        print("-" * 40)
        
        try:
            # 1. Criar base de vendas para o SKU
            print(" Criando base de vendas...")
            Venda = Base_venda(sku)
            
            if Venda is None or len(Venda) == 0:
                print(f"  SKU {sku} sem dados de venda. Pulando...")
                skus_com_erro += 1
                continue
            
            print(f"   Registros encontrados: {len(Venda)}")
            print(f"   Período: {Venda.index.min().date()} a {Venda.index.max().date()}")
            
            # 2. Executar modelo de validação cruzada
            print(" Executando modelo de validação cruzada...")
            resultados_modelo = modelo_validacao_cruzada_series_temporais(
                Venda, sku,'Log_Preco', 'Black_Friday', 'promocionado_25','Quarta-feira', 'Terça-feira',var_dpd = 'Log_Demanda',n_splits=n_splits
                )
            
            # 3. Fazer previsões para o SKU
            print(" Fazendo previsões...")
            resultado_previsao = prever_planilha(resultados_modelo, caminho_planilha_precos)
            
            # Adicionar SKU às previsões
            resultado_previsao['SKU_Modelo'] = sku
            
            # 4. Consolidar resultados
            resultado_sku = {
                'SKU': sku,
                'Data_Processamento': datetime.now(),
                'Registros_Base': len(Venda),
                'Intercepto': resultados_modelo['intercepto'],
                'Coeficiente_Preco': resultados_modelo['coeficientes'][0],  # Log_Preco
                'R2_Medio': resultados_modelo['metricas_medias']['r2'],
                'RMSE_Log': resultados_modelo['metricas_medias']['rmse'],
                'WAPE_Log': resultados_modelo['metricas_medias']['wape'],
                'TWAPE_Log': resultados_modelo['metricas_medias']['twape'],
                'RMSE_Original': resultados_modelo.get('rmse_original', None),
                'WAPE_Original': resultados_modelo.get('wape_original', None),
                'R2_Original': resultados_modelo.get('r2_original', None),
                'Erro_Medio': resultados_modelo['metricas_medias']['erro_medio'],
                'Status': 'Sucesso'
            }
            
            # Adicionar coeficientes dos dias da semana se existirem
            if len(resultados_modelo['coeficientes']) > 1:
                for i, coef in enumerate(resultados_modelo['coeficientes'][1:], 1):
                    resultado_sku[f'Coeficiente_Dia_{i}'] = coef
            
            resultados_consolidados.append(resultado_sku)
            previsoes_consolidadas.append(resultado_previsao)
            
            skus_processados += 1
            print(f" SKU {sku} processado com sucesso!")
            
        except Exception as e:
            print(f" ERRO no SKU {sku}: {str(e)}")
            print(traceback.format_exc())
            
            # Registrar erro
            resultado_erro = {
                'SKU': sku,
                'Data_Processamento': datetime.now(),
                'Registros_Base': 0,
                'Intercepto': None,
                'Coeficiente_Preco': None,
                'R2_Medio': None,
                'RMSE_Log': None,
                'WAPE_Log': None,
                'TWAPE_Log': None,
                'RMSE_Original': None,
                'WAPE_Original': None,
                'R2_Original': None,
                'Erro_Medio': None,
                'Status': f'Erro: {str(e)}'
            }
            
            resultados_consolidados.append(resultado_erro)
            skus_com_erro += 1
            continue
    
    # 5. Consolidar todos os resultados
    print(f"\n CONSOLIDANDO RESULTADOS...")
    print("=" * 60)
    
    df_resultados = pd.DataFrame(resultados_consolidados)
    
    if previsoes_consolidadas:
        df_previsoes = pd.concat(previsoes_consolidadas, ignore_index=True)
    else:
        df_previsoes = pd.DataFrame()
    
    # 6. Salvar resultados
    print(" Salvando arquivos...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar resultados dos modelos
    caminho_resultados = f'Resultados_Modelos_{timestamp}.xlsx'
    df_resultados.to_excel(caminho_resultados, index=False)
    
    # Salvar previsões consolidadas
    if not df_previsoes.empty:
        caminho_previsoes = f'Previsoes_Consolidadas_{timestamp}.xlsx'
        df_previsoes.to_excel(caminho_previsoes, index=False)
    
    print("=" * 60)
    print(" PIPELINE CONCLUÍDA!")
    print(f" SKUs processados com sucesso: {skus_processados}")
    print(f" SKUs com erro: {skus_com_erro}")
    print(f" Resultados salvos em: {caminho_resultados}")
    
    if not df_previsoes.empty:
        print(f" Previsões salvas em: {caminho_previsoes}")
    
    return df_resultados, df_previsoes

def processar_skus_em_lote(lista_skus, caminho_planilha_precos, n_splits=10):
    """
    Versão alternativa para processar lista de SKUs diretamente
    """
    df_produtos = pd.DataFrame({'ID_Sku': lista_skus})
    return pipeline_completa_skus(df_produtos, caminho_planilha_precos, n_splits)