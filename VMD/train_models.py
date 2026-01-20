import os
import sys
import pandas as pd
import warnings
import json
from statsmodels.iolib.smpickle import save_pickle
from joblib import Parallel, delayed
import multiprocessing

# --- 1. Configurar o Caminho do Projeto ---
# Define o caminho raiz do projeto de forma robusta
try:
    # Em modo de script, __file__ está definido
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
except NameError:
    # Em modo interativo, __file__ não está definido, usa o diretório atual
    # Certifique-se de que o terminal/notebook está na raiz do projeto
    project_root = os.getcwd()

if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. Importar Funções Personalizadas ---
from Functions.FNC_Pro import configurar_credenciais_bq, Base_venda
from Functions.FNC_SARIMAX import encontrar_melhores_parametros_sarimax, modelo_sarimax

warnings.filterwarnings("ignore")

def treinar_modelo_sku(sku, modelos_output_dir, exog_vars):
    """
    Função de worker para treinar e salvar o modelo para um único SKU.
    """
    # Esta função será executada em um processo separado para cada SKU.
    # print(f"--- Processando SKU: {sku} ---")
    
    df_venda = Base_venda(sku)

    if df_venda is None or df_venda.empty or len(df_venda) < 60:
        # Retorna o status para o sumário final, sem poluir o log.
        return sku, "Dados insuficientes"

    try:
        best_order, best_seasonal_order, trend = encontrar_melhores_parametros_sarimax(
            df_venda, sku, exog_vars, verbose=False # Verbose desativado para saídas mais limpas
        )
    except Exception as e:
        print(f"SKU {sku}: Erro fatal ao encontrar parâmetros: {e}")
        return sku, f"Erro nos parâmetros: {e}"
        
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
            
            # print(f"SKU {sku}: Modelo e parâmetros salvos com sucesso.")
            return sku, "Sucesso"

        except Exception as e:
            print(f"SKU {sku}: Erro ao salvar o modelo ou parâmetros: {e}")
            return sku, f"Erro ao salvar: {e}"
    else:
        print(f"SKU {sku}: Falha ao treinar o modelo.")
        return sku, "Falha no treinamento"

def pipeline_treinamento_de_modelos(lista_skus, credenciais_path, modelos_output_dir):
    """
    Executa o pipeline de treinamento de forma paralela para acelerar o processo.
    """
    print("--- Iniciando Pipeline de TREINAMENTO de Modelos VMD ---")
    
    # --- Configuração Inicial ---
    # As credenciais são configuradas uma vez no processo principal.
    # Processos filhos herdarão o estado (ex: variáveis de ambiente).
    configurar_credenciais_bq(credenciais_path)
    os.makedirs(modelos_output_dir, exist_ok=True)
    print(f"Modelos e parâmetros serão salvos em: {modelos_output_dir}")

    exog_vars = [
        'Log_Preco', 'Black_Friday', 'promocionado',
        'Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 
        'Sexta-feira', 'Sábado', 'Domingo'
    ]

    # --- Execução Paralela do Treinamento ---
    num_cores = multiprocessing.cpu_count()
    print(f"Utilizando {num_cores} núcleos para treinar {len(lista_skus)} SKUs em paralelo.")

    resultados = Parallel(n_jobs=num_cores, backend="threading")(
        delayed(treinar_modelo_sku)(sku, modelos_output_dir, exog_vars) for sku in lista_skus
    )

    # --- Sumário do Treinamento ---
    print("\n--- Sumário do Treinamento ---")
    resultados_validos = [r for r in resultados if r is not None]
    sucessos = [r for r in resultados_validos if r[1] == "Sucesso"]
    falhas = [r for r in resultados_validos if r[1] != "Sucesso"]
    
    print(f"Total de SKUs para processar: {len(lista_skus)}")
    print(f"Modelos treinados com sucesso: {len(sucessos)}")
    print(f"SKUs com falha ou pulados: {len(falhas)}")
    
    if falhas:
        print("\nDetalhamento das falhas:")
        # Agrupar SKUs por motivo da falha para um relatório mais limpo
        falhas_agrupadas = {}
        for sku, motivo in falhas:
            if motivo not in falhas_agrupadas:
                falhas_agrupadas[motivo] = []
            falhas_agrupadas[motivo].append(sku)
        
        for motivo, skus_falha in sorted(falhas_agrupadas.items()):
            print(f"  - Motivo: {motivo}")
            print(f"    SKUs ({len(skus_falha)}): {skus_falha}")

    print("\n--- Pipeline de TREINAMENTO Concluído ---")

if __name__ == '__main__':
    caminho_credenciais = r'G:/Drives compartilhados/Bases BI/epoca-230913-b478a9a0dd4c.json'
    skus_para_treinar = ['88264','52774']

    # A raiz do projeto já foi definida no início do script
    # Pasta para salvar os modelos, dentro da raiz do projeto
    pasta_modelos = os.path.join(project_root, 'Modelos_VMD')

    pipeline_treinamento_de_modelos(skus_para_treinar, caminho_credenciais, pasta_modelos)