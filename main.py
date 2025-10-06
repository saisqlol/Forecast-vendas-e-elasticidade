import os
import sys
import pandas as pd
from google.cloud import storage

# --- Configuração do Caminho do Projeto ---

current_dir = os.getcwd()
project_root = current_dir 
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Importações do Projeto ---
from Functions.Pipeline import pipeline_completa_skus
from Functions.FNC_Pro import lista_produtos, configurar_credenciais_bq
from config import (
    BQ_CREDENTIALS_PATH,
    PRODUCT_INFO_PATH,
    CUSTOM_SKU_LIST_PATH,
    N_SPLITS_TSCV,
    GCS_BUCKET_NAME,
    GCS_BLOB_NAME
)

def main():
    """
    Função principal para executar o pipeline de treinamento de modelos de forecast.
    """
    print("--- INICIANDO EXECUÇÃO DO PIPELINE DE TREINAMENTO ---")
    
    # --- 1. Configurar Credenciais ---
    try:
        configurar_credenciais_bq(BQ_CREDENTIALS_PATH)
        print("Credenciais do Google Cloud configuradas com sucesso.")
    except Exception as e:
        print(f"ERRO: Falha ao configurar credenciais do GCP. Verifique o caminho em config.py. Detalhes: {e}")
        return

    # --- 2. Selecionar SKUs para Treinamento ---
    # Lógica de decisão: se um arquivo de lista personalizada existir, use-o.
    # Caso contrário, use os filtros padrão.
    
    use_custom_list = True  # Mude para False para usar os filtros de Classificação/Ativo
    
    if use_custom_list and os.path.exists(CUSTOM_SKU_LIST_PATH):
        print(f"Modo de execução: Lista de SKUs personalizada encontrada em '{CUSTOM_SKU_LIST_PATH}'.")
        try:
            df_custom_skus = pd.read_excel(CUSTOM_SKU_LIST_PATH)
            sku_list = df_custom_skus.iloc[:, 0].astype(str).tolist()
            produtos = lista_produtos(PRODUCT_INFO_PATH, SKUS=sku_list)
        except Exception as e:
            print(f"ERRO: Falha ao ler a lista de SKUs personalizada. Detalhes: {e}")
            return
    else:
        print("Modo de execução: Filtros padrão (Classificação='A', Ativo='Sim').")
        produtos = lista_produtos(
            PRODUCT_INFO_PATH,
            Classificacao='A',
            Ativo='Sim'
        )

    if produtos.empty:
        print("Nenhum produto encontrado para processar. Encerrando o pipeline.")
        return

    # --- 3. Executar Pipeline de Treinamento ---
    resultados_consolidados = pipeline_completa_skus(
        df_produtos=produtos,
        n_splits=N_SPLITS_TSCV
    )

    if resultados_consolidados is None or resultados_consolidados.empty:
        print("Pipeline não retornou resultados. Encerrando.")
        return

    # --- 4. Salvar e Fazer Upload dos Resultados ---
    print("\n--- SALVANDO RESULTADOS ---")
    local_filename = "Resultados_Elasticidade.csv"
    try:
        resultados_consolidados.to_csv(local_filename, index=False, sep=';', decimal=',')
        print(f"Arquivo '{local_filename}' salvo localmente.")

        print(f"Fazendo upload para o Google Cloud Storage em gs://{GCS_BUCKET_NAME}/{GCS_BLOB_NAME}...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_BLOB_NAME)
        blob.upload_from_filename(local_filename)
        print("Upload concluído com sucesso.")

    except Exception as e:
        print(f"ERRO: Falha ao salvar ou fazer upload dos resultados. Detalhes: {e}")
    finally:
        # Limpar o arquivo local após a tentativa de upload
        if os.path.exists(local_filename):
            os.remove(local_filename)
            print(f"Arquivo local '{local_filename}' removido.")

    print("\n--- PIPELINE CONCLUÍDO ---")

if __name__ == "__main__":
    main()

