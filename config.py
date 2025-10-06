# config.py

# --- Configurações do Google Cloud ---
GCP_PROJECT_ID = "epoca-230913"
# No Docker, este caminho será o local do arquivo de credenciais dentro do contêiner
BQ_CREDENTIALS_PATH = "gcp-credentials.json" 
GCS_BUCKET_NAME = "epoca-storage"
GCS_BLOB_NAME = "senior-estoque/Resultados Modelos/Forecast/Resultados_Elasticidade.csv"

# --- Caminhos de Dados ---
# Caminho para o arquivo principal que contém as informações dos produtos
PRODUCT_INFO_PATH = "G:/Drives compartilhados/Planilha de Impostos/BI/Planilha com Impostos.xlsm"
# Caminho para uma lista opcional de SKUs específicos para treinar
CUSTOM_SKU_LIST_PATH = "Lista Produtos/Lista__.xlsx"

# --- Parâmetros do Modelo ---
N_SPLITS_TSCV = 10

# Caminhos de Saída
MODELS_DIR = "../Modelos"
RESULTS_DIR = "../Resultados"