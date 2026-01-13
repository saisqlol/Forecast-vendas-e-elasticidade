from google.cloud import bigquery
import pandas as pd
import os
import numpy as np


# Pega a lista dos produtos top 10 em cada categoria 

def best_sellers(base_selecionada):
    df = pd.read_excel(base_selecionada)
    df = df[['SKU']]

    return df

def produtos_selecionados(base_selecionada):
    df = pd.read_excel(base_selecionada)
    lista_de_valores = df.iloc[:, 0].astype(str).tolist()

    return lista_de_valores

def lista_produtos(base_produtos, Classificacao=None, Ativo=None, SKUS=None):
    """
    Lê a lista de produtos de uma planilha Excel, com filtros opcionais.

    Args:
        base_produtos (str): Caminho para o arquivo Excel.
        Classificacao (str, optional): Filtra pela classificação ('A', 'B', 'C'). Defaults to None.
        Ativo (str, optional): Filtra por status ('Sim' ou 'Não'). Defaults to None.
        SKUS (list, optional): Filtra por uma lista específica de SKUs. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame com a coluna 'ID_Sku' dos produtos filtrados.
    """
    # --- Processar a aba 'PLAN DE MKT (2)' ---
    df = pd.read_excel(base_produtos, 'PLAN DE MKT (2)')
    
    # Aplicar filtro de Classificação, se fornecido (usando 'contains')
    if Classificacao and 'Classificação 2.0' in df.columns:
        df = df[df['Classificação 2.0'].astype(str).str.contains(Classificacao, case=False, na=False)]
        
    df = df[['ID_Sku']]

    # --- Processar a aba 'KITs Virtuais' ---
    df2 = pd.read_excel(base_produtos, 'KITs Virtuais')

    # Aplicar filtro de Classificação, se fornecido (usando 'contains')
    if Classificacao and 'Classificação SKU' in df2.columns:
        df2 = df2[df2['Classificação SKU'].astype(str).str.contains(Classificacao, case=False, na=False)]

    # Aplicar filtro de Ativo, se fornecido
    if Ativo and 'Sku_Ativo' in df2.columns:
        # Converte a coluna e o filtro para minúsculas para uma comparação robusta
        df2 = df2[df2['Sku_Ativo'].astype(str).str.strip().str.lower() == Ativo.lower()]

    # Renomear e selecionar a coluna do SKU
    if 'ID_KIT' in df2.columns:
        df2 = df2[['ID_KIT']].rename(columns={'ID_KIT': 'ID_Sku'})
    else:
        df2 = pd.DataFrame(columns=['ID_Sku'])

    # --- Consolidar os resultados ---
    dff = pd.concat([df, df2], ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    # Aplicar filtro de SKUs específicos, se fornecido
    if SKUS:
        # Garantir que os SKUs na lista e no DataFrame sejam do mesmo tipo (string) para a comparação
        skus_a_filtrar = [str(s) for s in SKUS]
        dff['ID_Sku'] = dff['ID_Sku'].astype(str)
        dff = dff[dff['ID_Sku'].isin(skus_a_filtrar)]
    
    print(f"Encontrados {len(dff)} SKUs com os filtros: Classificação='{Classificacao}', Ativo='{Ativo}', SKUs='{SKUS if SKUS else 'Todos'}'")
    
    return dff
    
    

def configurar_credenciais_bq(arquivo_json):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = arquivo_json


def Base_venda(sku):
    query =f"""
    SELECT Data, SKU, Preco_Listado,  Preco,  Demanda
    FROM `epoca-230913.VTEX.Extracao_Base_Modelo_VMD`  
    WHERE sku = '{sku}' AND Data >= '2024-09-22' AND Venda_Considerar = 1
    """
    client = bigquery.Client()
    query_job = client.query(query)
    df= query_job.result().to_dataframe()
    
    if df.empty:
        print(f"SKU {sku} sem dados retornados do BigQuery")
        return pd.DataFrame()
    
    # Converter Data
    df['Data'] = pd.to_datetime(df['Data'])
    df['Preco'] = df['Preco'].astype(float)
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

    # Substituir -inf e NaN por 0 após o cálculo dos logs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna({'Log_Preco_7D': 0, 'Log_Demanda_7D': 0}, inplace=True)

    condicao = (df['Log_Demanda'] == 0) | (np.isinf(df['Log_Demanda']))
    df.loc[condicao, 'Log_Demanda'] = df.loc[condicao, 'Log_Demanda_7D']
    
    # Criar a coluna 'promocionado' com base no desvio do preço em relação à média móvel de 7 dias
    # É considerado promocionado se o preço for 10% ou mais abaixo da média dos últimos 7 dias.
    df['promocionado'] = (df['Preco'] <= df['Med_Preco_7_Dia'] * 0.90).astype(int)
    
    df = df.drop(columns=['Dia_Semana','Med_Preco_7_Dia','Med_Demanda_7_Dia'])

    # 1. Filtrar a partir da primeira data com venda para garantir que a série comece com atividade
    if not df.empty and (df['Demanda'] > 0).any():
        primeira_venda_data = df[df['Demanda'] > 0].index.min()
        df = df[df.index >= primeira_venda_data]
        print(f"SKU {sku}: Histórico de dados ajustado para começar em {primeira_venda_data.date()}, o primeiro dia com vendas.")
    
    # 2. Se, após o ajuste, mais de 50% dos dados da coluna Demanda for igual a 0, filtra o df
    # Comentado em 06/01/2026: Esta lógica é muito agressiva para o cálculo de VMD,
    # pois VMD deve considerar os dias com venda zero na média. Removê-los distorce o resultado.
    # if not df.empty and (df['Demanda'] == 0).sum() / len(df) > 0.5:
    #     print(f"SKU {sku}: Mais de 50% da demanda restante é 0. Filtrando para manter apenas dias com vendas.")
    #     df = df[df['Demanda'] != 0]
        
    return df

    # print(f"SKU {sku} sem dados retornados do BigQuery")
    # return pd.DataFrame(), False

    # --- Agregação para garantir datas únicas ---
    # Converte a data e agrupa os dados para garantir que cada dia seja uma única linha.
    # Isso resolve o problema de 'duplicate labels' ao somar a demanda e tirar a média do preço.
    df['Data'] = pd.to_datetime(df['Data'])
    if df.duplicated(subset=['Data']).any():
        print(f"SKU {sku}: Encontradas múltiplas entradas para a mesma data. Agregando os dados diários.")
        df = df.groupby(pd.Grouper(key='Data', freq='D')).agg({
            'SKU': 'first',
            'Preco_Listado': 'mean',
            'Preco': 'mean',
            'Demanda': 'sum',
            'Estoque': 'last',
            'Ruptura': 'last',
            'Venda_Considerar': 'max'
        }).reset_index()
        # O groupby pode criar NaNs para SKUs em dias agregados sem dados, preenchendo-os.
        df['SKU'] = df['SKU'].fillna(method='ffill').fillna(method='bfill')
    
    # --- Conversão de Tipos e Ordenação ---
    df['Data'] = pd.to_datetime(df['Data'])
    df['Demanda'] = df['Demanda'].astype(float)
    df['Ruptura'] = pd.to_numeric(df['Ruptura'], errors='coerce').fillna(0).astype(int)
    df['Venda_Considerar'] = pd.to_numeric(df['Venda_Considerar'], errors='coerce').fillna(0).astype(int)
    df.sort_values('Data', inplace=True)
    
    # --- Feature Engineering e Processamento ---
    df['Log_Preco'] = np.log(df['Preco'].clip(lower=0.01))
    df['Log_Demanda'] = np.log(df['Demanda'].clip(lower=0.01))
    
    df['Med_Demanda_7_Dia'] = df['Demanda'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['Log_Demanda_7D'] = np.log(df['Med_Demanda_7_Dia'].clip(lower=0.01))
    
    condicao_problema = np.isinf(df['Log_Demanda']) | np.isnan(df['Log_Demanda'])
    if condicao_problema.any():
        df.loc[condicao_problema, 'Log_Demanda'] = df.loc[condicao_problema, 'Log_Demanda_7D']
    
    df.set_index('Data', inplace=True)
    df.sort_index(inplace=True)

    df['Dia_Semana'] = df.index.day_name(locale='pt_BR')
    df['Black_Friday'] = ((df.index.month == 11) & (df.index.day >= 28) & (df.index.day <= 30)).astype(int)
    
    dias_da_semana_dummies = pd.get_dummies(df['Dia_Semana'], dtype=int)
    df = pd.concat([df, dias_da_semana_dummies], axis=1)

    df['Med_Preco_7_Dia'] = df['Preco'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['promocionado'] = (df['Preco'] <= df['Med_Preco_7_Dia'] * 0.90).astype(int)
    
    colunas_para_remover = [
        'Dia_Semana', 'Med_Preco_7_Dia', 'Med_Demanda_7_Dia', 'Log_Demanda_7D',
        'Estoque', 'Ruptura', 'Venda_Considerar'
    ]

    # Remover colunas auxiliares
    df = df.drop(columns=[col for col in colunas_para_remover if col in df.columns])

    # Comentado em 08/01/2026: Esta lógica removia os dias com venda zero do início do período,
    # o que distorcia o cálculo do VMD (ex: dividia por 50 dias em vez de 60).
    # O VMD correto deve sempre considerar o período completo.
    # if not df.empty and (df['Demanda'] > 0).any():
    #     primeira_venda_data = df[df['Demanda'] > 0].index.min()
    #     df = df[df.index >= primeira_venda_data]
    #     # print(f"SKU {sku}: Histórico de dados ajustado para começar em {primeira_venda_data.date()}, o primeiro dia com vendas.")

    return df

