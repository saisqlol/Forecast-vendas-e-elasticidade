from google.cloud import bigquery
import pandas as pd
import os
import numpy as np


def lista_produtos(base_produtos):
    df = pd.read_excel(base_produtos,'PLAN DE MKT (2)')
    df = df[['ID_Sku']]
    df2 = pd.read_excel(base_produtos,'KITs Virtuais')
    df2 = df2[['ID_KIT']].drop_duplicates()
    df2 = df2.rename(columns={'ID_KIT':'ID_Sku'})
    dff = pd.concat([df,df2],ignore_index=True)
    return dff
    
    

def configurar_credenciais_bq(arquivo_json):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = arquivo_json


def Base_venda(sku):
    query =f"""
    SELECT Data, SKU, Preco_Listado, Med_Preco_Dia AS Preco, Qtd_Vendida AS Demanda 
    FROM `epoca-230913.VTEX.Extracao_Base_Modelo` 
    WHERE sku = '{sku}'
    """
    client = bigquery.Client()
    query_job = client.query(query)
    df= query_job.result().to_dataframe()
    
    if df.empty:
        print(f"SKU {sku} sem dados retornados do BigQuery")
        return pd.DataFrame()
    
    # Converter Data
    df['Data'] = pd.to_datetime(df['Data'])
    df['Demanda'] = df['Demanda'].astype(float)

    # Remover duplicatas de índice
    duplicatas = df.index.duplicated()
    if duplicatas.any():
        print(f"Encontradas {duplicatas.sum()} linhas com índices duplicados. Removendo...")
        df = df[~duplicatas]
    
    # Calcular logs com tratamento robusto
    df['Log_Preco'] = np.log(df['Preco'].clip(lower=0.01))  # Evitar log(0)
    df['Log_Demanda'] = np.log(df['Demanda'].clip(lower=0.01))  # Evitar log(0)
    
    # Calcular médias móveis para imputação
    df['Med_Demanda_7_Dia'] = (
        df.groupby('SKU')['Demanda']
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    )
    df['Log_Demanda_7D'] = np.log(df['Med_Demanda_7_Dia'].clip(lower=0.01))
    
    # Verificar e corrigir valores problemáticos
    condicao_problema = np.isinf(df['Log_Demanda']) | np.isnan(df['Log_Demanda']) | (df['Log_Demanda'] < -20)
    if condicao_problema.any():
        print(f"Corrigindo {condicao_problema.sum()} valores problemáticos em Log_Demanda")
        df.loc[condicao_problema, 'Log_Demanda'] = df.loc[condicao_problema, 'Log_Demanda_7D']
    
    # Remover quaisquer valores problemáticos remanescentes
    condicao_final = np.isinf(df['Log_Demanda']) | np.isnan(df['Log_Demanda'])
    if condicao_final.any():
        print(f"Removendo {condicao_final.sum()} valores ainda problemáticos")
        df = df[~condicao_final]
    
    df['Ano_Mes'] = df['Data'].dt.strftime('%Y-%m')
    df['Dia_Semana'] = df['Data'].dt.day_name(locale='pt_BR')
    # Coluna Black Friday (entre 28 e 30 de novembro)
    df['Black_Friday'] = ((df['Data'].dt.month == 11) & 
                          (df['Data'].dt.day >= 28) & 
                          (df['Data'].dt.day <= 30)).astype(int)
    # Definir indice
    df.set_index('Data', inplace=True)
    df.sort_index(inplace=True)

    duplicatas = df.index.duplicated()
    if duplicatas.any():
        print(f"Encontradas {duplicatas.sum()} linhas com índices duplicados. Removendo...")
        df = df[~duplicatas]

    dias_da_semana_dummies = pd.get_dummies(df['Dia_Semana'], dtype=int)
    df = pd.concat([df, dias_da_semana_dummies], axis=1)
    df['Med_Preco_7_Dia'] = (
        df.groupby('SKU')['Preco']
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    )
    df['Med_Demanda_7_Dia'] = (
        df.groupby('SKU')['Demanda']
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    )

    df['Log_Preco_7D'] = np.log(df['Med_Preco_7_Dia'])
    df['Log_Demanda_7D'] = np.log(df['Med_Demanda_7_Dia'])
    condicao = (df['Log_Demanda'] == 0) | (np.isinf(df['Log_Demanda']))
    df.loc[condicao, 'Log_Demanda'] = df.loc[condicao, 'Log_Demanda_7D']
    # Calcular desconto percentual
    df['Desconto_Percentual'] = (df['Preco_Listado'] - df['Preco']) / df['Preco_Listado'] * 100
    # Coluna promocionado_25 (desconto >= 25%)
    df['promocionado_25'] = (df['Desconto_Percentual'] >= 25).astype(int)
    # Coluna promocionado_50 (desconto >= 50%)
    df['promocionado_50'] = (df['Desconto_Percentual'] >= 50).astype(int)
    df = df.drop(columns=['Dia_Semana','Med_Preco_7_Dia','Med_Demanda_7_Dia','Desconto_Percentual'])

    # Se mais de 50% dos dados da coluna Demanda for igual a 0, filtra o df
    if not df.empty and (df['Demanda'] == 0).sum() / len(df) > 0.5:
        print(f"SKU {sku}: Mais de 50% da demanda é 0. Filtrando para manter apenas dias com vendas.")
        df = df[df['Demanda'] != 0]
        
    return df

