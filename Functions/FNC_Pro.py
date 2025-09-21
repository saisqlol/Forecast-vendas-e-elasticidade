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
    SELECT Data, SKU,Preco_Listado, Med_Preco_Dia AS Preco, Qtd_Vendida AS Demanda FROM `epoca-230913.VTEX.Extracao_Base_Modelo` WHERE sku = '{sku}'
    """
    client = bigquery.Client()
    query_job = client.query(query)
    df= query_job.result().to_dataframe()
    df['Log_Preco_Listado'] = np.log(df['Preco_Listado'])
    df['Log_Preco'] = np.log(df['Preco'])
    df['Log_Demanda'] = np.log(df['Demanda'])
    df['Data'] = pd.to_datetime(df['Data'])
    df['Dia_Semana'] = df['Data'].dt.day_name(locale='pt_BR')
    # Coluna Black Friday (entre 28 e 30 de novembro)
    df['Black_Friday'] = ((df['Data'].dt.month == 11) & 
                          (df['Data'].dt.day >= 28) & 
                          (df['Data'].dt.day <= 30)).astype(int)
    df.set_index('Data', inplace=True)
    df.sort_index(inplace=True)
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
    df.loc[condicao, 'Log_Demanda'] = df['Log_Demanda_7D']
    # Calcular desconto percentual
    df['Desconto_Percentual'] = (df['Preco_Listado'] - df['Preco']) / df['Preco_Listado'] * 100
    # Coluna promocionado_25 (desconto >= 25%)
    df['promocionado_25'] = (df['Desconto_Percentual'] >= 25).astype(int)
    # Coluna promocionado_50 (desconto >= 50%)
    df['promocionado_50'] = (df['Desconto_Percentual'] >= 50).astype(int)
    df = df.drop(columns=['Dia_Semana','Med_Preco_7_Dia','Med_Demanda_7_Dia','Desconto_Percentual'])
    return df

