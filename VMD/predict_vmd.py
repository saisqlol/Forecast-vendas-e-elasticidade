import os
import sys
import pandas as pd
import warnings
import json
from statsmodels.iolib.smpickle import load_pickle

# --- 1. Configurar o Caminho do Projeto ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. Importar Funções Personalizadas ---
from Functions.FNC_Pro import configurar_credenciais_bq, Base_venda
from Functions.FNC_SARIMAX import gerar_previsoes_vmd

warnings.filterwarnings("ignore")
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 50)

def pipeline_previsao_vmd(lista_skus, credenciais_path, modelos_input_dir):
    """
    Carrega modelos e parâmetros já treinados para gerar previsões de VMD.
    """
    print("--- Iniciando Pipeline de PREVISÃO de VMD ---")
    
    # --- Configuração Inicial ---
    configurar_credenciais_bq(credenciais_path)
    
    resultados_finais = []

    # --- Loop de Previsão por SKU ---
    for i, sku in enumerate(lista_skus):
        print(f"\n--- Gerando Previsão para SKU {i+1}/{len(lista_skus)}: {sku} ---")

        model_path = os.path.join(modelos_input_dir, f'model_sku_{sku}.pkl')
        params_path = os.path.join(modelos_input_dir, f'params_sku_{sku}.json')

        # Verificar se o modelo e os parâmetros existem
        if not os.path.exists(model_path) or not os.path.exists(params_path):
            print(f"Modelo ou parâmetros não encontrados para o SKU {sku}. Pule este SKU.")
            resultados_finais.append({'SKU': sku, 'Status': 'Modelo não treinado'})
            continue

        try:
            # Carregar o modelo e os parâmetros
            modelo_carregado = load_pickle(model_path)
            with open(params_path, 'r') as f:
                parametros = json.load(f)
            
            exog_vars = parametros['exog_vars']
            print(f"Modelo e parâmetros para SKU {sku} carregados com sucesso.")

            # Obter os dados mais recentes
            df_venda = Base_venda(sku)

            if df_venda is None or df_venda.empty:
                print(f"Não foi possível obter dados de venda para o SKU {sku}.")
                resultados_finais.append({'SKU': sku, 'Status': 'Erro ao obter dados'})
                continue

            # Gerar as previsões de VMD
            previsoes = gerar_previsoes_vmd(modelo_carregado, df_venda, sku, exog_vars)
            resultados_finais.append(previsoes)
            print(f"Previsão para SKU {sku} gerada.")

        except Exception as e:
            print(f"Ocorreu um erro ao gerar previsão para o SKU {sku}: {e}")
            resultados_finais.append({'SKU': sku, 'Status': f'Erro: {e}'})

    print("\n--- Pipeline de PREVISÃO Concluído ---")
    return pd.DataFrame(resultados_finais)


if __name__ == '__main__':
    # O caminho correto para o arquivo de credenciais.
    caminho_credenciais = r'G:/Drives compartilhados/Bases BI/epoca-230913-b478a9a0dd4c.json'
    
    skus_para_prever = ['88264','52774','83626','26619','25357','30713','36954','48639','22852','36950','76175','36947','25317','129572','4012','36949','55454','110006','30712','23756']

    # --- Define os caminhos de forma robusta para funcionar em modo script e interativo ---
    try:
        # Modo Script: __file__ está definido
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_main = os.path.abspath(os.path.join(current_script_dir, '..'))
    except NameError:
        # Modo Interativo: __file__ não está definido
        project_root_main = os.getcwd()
    
    pasta_modelos = os.path.join(project_root_main, 'Modelos_VMD')

    # Executar o pipeline de previsão
    df_previsoes = pipeline_previsao_vmd(skus_para_prever, caminho_credenciais, pasta_modelos)

    # Exibir os resultados
    print("\n--- Resultados Finais da Previsão ---")
    print(df_previsoes)




df_previsoes.to_csv(os.path.join(project_root_main, 'previsoes_vmd.csv'), index=False)