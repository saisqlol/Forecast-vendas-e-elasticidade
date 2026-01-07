import os
import sys
import pandas as pd
import warnings
import json
from statsmodels.iolib.smpickle import save_pickle

# --- 1. Configurar o Caminho do Projeto ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. Importar Funções Personalizadas ---
from Functions.FNC_Pro import configurar_credenciais_bq, Base_venda
from Functions.FNC_SARIMAX import encontrar_melhores_parametros_sarimax, modelo_sarimax

warnings.filterwarnings("ignore")

def pipeline_treinamento_de_modelos(lista_skus, credenciais_path, modelos_output_dir):
    """
    Executa o pipeline de treinamento, encontra os melhores parâmetros e salva
    os modelos e os parâmetros para cada SKU.
    """
    print("--- Iniciando Pipeline de TREINAMENTO de Modelos VMD ---")
    
    # --- Configuração Inicial ---
    configurar_credenciais_bq(credenciais_path)
    os.makedirs(modelos_output_dir, exist_ok=True)
    print(f"Modelos e parâmetros serão salvos em: {modelos_output_dir}")

    exog_vars = [
        'Log_Preco', 'Black_Friday', 'promocionado',
        'Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 
        'Sexta-feira', 'Sábado', 'Domingo'
    ]

    # --- Loop de Treinamento por SKU ---
    for i, sku in enumerate(lista_skus):
        print(f"\n--- Treinando Modelo para SKU {i+1}/{len(lista_skus)}: {sku} ---")
        
        df_venda = Base_venda(sku)

        if df_venda is None or df_venda.empty or len(df_venda) < 60:
            print(f"Dados insuficientes para o SKU {sku}. Pulando.")
            continue

        try:
            best_order, best_seasonal_order, trend = encontrar_melhores_parametros_sarimax(
                df_venda, sku, exog_vars, verbose=True # Ativar verbose para análise
            )
        except Exception as e:
            print(f"Erro fatal ao encontrar parâmetros para SKU {sku}: {e}")
            continue
            
        resultado_modelo = modelo_sarimax(
            df_venda, sku, *exog_vars, 
            order=best_order, 
            seasonal_order=best_seasonal_order, 
            trend=trend,
            verbose=False
        )

        if resultado_modelo:
            try:
                # Salvar os PARÂMETROS em um arquivo JSON
                parametros = {
                    'order': best_order,
                    'seasonal_order': best_seasonal_order,
                    'trend': trend,
                    'exog_vars': exog_vars
                }
                params_path = os.path.join(modelos_output_dir, f'params_sku_{sku}.json')
                with open(params_path, 'w') as f:
                    json.dump(parametros, f)
                
                # Salvar o MODELO treinado em um arquivo .pkl
                model_path = os.path.join(modelos_output_dir, f'model_sku_{sku}.pkl')
                save_pickle(resultado_modelo, model_path)
                
                print(f"Modelo e parâmetros para o SKU {sku} salvos com sucesso.")

            except Exception as e:
                print(f"Erro ao salvar o modelo ou parâmetros para o SKU {sku}: {e}")
        else:
            print(f"Falha ao treinar o modelo para o SKU {sku}.")

    print("\n--- Pipeline de TREINAMENTO Concluído ---")

if __name__ == '__main__':
    caminho_credenciais = r'G:/Drives compartilhados/Bases BI/epoca-230913-b478a9a0dd4c.json'
    skus_para_treinar = ['88264','52774','83626','26619','25357','30713','36954','48639','22852','36950','76175','36947','25317','129572','4012','36949','55454','110006','30712','23756']

    # --- Define os caminhos de forma robusta para funcionar em modo script e interativo ---
    try:
        # Modo Script: __file__ está definido. Pega o caminho do script e sobe um nível.
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_main = os.path.abspath(os.path.join(current_script_dir, '..'))
    except NameError:
        # Modo Interativo: __file__ não está definido. Usa o diretório de trabalho atual (CWD).
        # Garanta que seu terminal VS Code está na raiz do projeto para isso funcionar.
        project_root_main = os.getcwd()

    # Pasta para salvar os modelos, dentro da raiz do projeto
    pasta_modelos = os.path.join(project_root_main, 'Modelos_VMD')

    pipeline_treinamento_de_modelos(skus_para_treinar, caminho_credenciais, pasta_modelos)
