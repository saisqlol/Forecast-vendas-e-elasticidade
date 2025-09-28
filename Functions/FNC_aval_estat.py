import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler

def avaliar_dados_series_temporais(df, sku):
    """
    Avaliação estatística completa dos dados para modelagem de séries temporais
    """
    print(f"=== AVALIAÇÃO ESTATÍSTICA PARA SKU {sku} ===\n")
    
    # Criar dados mensais agregados para análise sazonal
    df_mensal = df.copy()
    df_mensal['AnoMes'] = df_mensal.index.strftime('%Y-%m')
    
    # Agregar por mês: soma da demanda e média do preço
    mensal_agg = df_mensal.groupby('AnoMes').agg({
        'Preco': 'mean',
        'Demanda': 'sum',
        'Log_Preco': 'mean',
        'Log_Demanda': 'mean'
    }).reset_index()
    
    # 1. Análise descritiva básica
    print("1. ESTATÍSTICAS DESCRITIVAS:")
    print("=" * 50)
    
    numeric_cols = ['Preco', 'Demanda', 'Log_Preco', 'Log_Demanda', 
                   'Log_Preco_7D', 'Log_Demanda_7D']
    
    desc_stats = df[numeric_cols].describe()
    print(desc_stats.round(4))
    print("\n")
    
    # 2. Verificação de valores missing
    print("2. VALORES MISSING:")
    print("=" * 50)
    missing = df.isnull().sum()
    print(missing)
    print("\n")
    
    # 3. Teste de estacionariedade (ADF Test) - CRÍTICO!
    print("3. TESTE DE ESTACIONARIEDADE (ADF TEST):")
    print("=" * 50)
    print("IMPORTANTE: Séries não estacionárias podem invalidar a modelagem!")
    print("Se p-valor > 0.05, a série NÃO é estacionária\n")
    
    resultados_estacionariedade = {}
    for col in ['Log_Demanda', 'Demanda']:
        result = adfuller(df[col].dropna())
        resultados_estacionariedade[col] = result[1] < 0.05
        
        print(f"{col}:")
        print(f"  Estatística ADF: {result[0]:.4f}")
        print(f"  p-valor: {result[1]:.4f}")
        if result[1] < 0.05:
            print("  → Série ESTACIONÁRIA (rejeita H0) - OK para modelagem")
        else:
            print("  → Série NÃO ESTACIONÁRIA (não rejeita H0) - Precisa de transformação!")
        print()
    
    # 4. Teste de normalidade (Shapiro-Wilk)
    print("4. TESTE DE NORMALIDADE (SHAPIRO-WILK):")
    print("=" * 50)
    
    for col in ['Log_Demanda', 'Log_Preco']:
        stat, p_value = stats.shapiro(df[col].dropna())
        print(f"{col}:")
        print(f"  Estatística: {stat:.4f}")
        print(f"  p-valor: {p_value:.4f}")
        if p_value > 0.05:
            print("  → Distribuição NORMAL (não rejeita H0)")
        else:
            print("  → Distribuição NÃO NORMAL (rejeita H0)")
        print()
    
    # 5. Correlação entre variáveis
    print("5. MATRIZ DE CORRELAÇÃO:")
    print("=" * 50)
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix.round(4))
    print("\n")
    
    # 6. Gráficos
    print("6. GERANDO GRÁFICOS...")
    print("=" * 50)
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 25))
    
    # 6.1 Série temporal original com média móvel
    plt.subplot(4, 2, 1)
    plt.plot(df.index, df['Preco'], 'b-', label='Preço', alpha=0.5)
    plt.plot(df.index, df['Preco'].rolling(7).mean(), 'darkblue', label='MM7 Preço', linewidth=2)
    plt.title('Série Temporal - Preço (com média móvel 7 dias)')
    plt.xlabel('Ano-Mês')
    plt.ylabel('Preço')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 2, 2)
    plt.plot(df.index, df['Demanda'], 'r-', label='Demanda', alpha=0.5)
    plt.plot(df.index, df['Demanda'].rolling(7).mean(), 'darkred', label='MM7 Demanda', linewidth=2)
    plt.title('Série Temporal - Demanda (com média móvel 7 dias)')
    plt.xlabel('Ano-Mês')
    plt.ylabel('Demanda')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6.2 Análise de sazonalidade por Ano-Mês
    plt.subplot(4, 2, 3)
    plt.bar(mensal_agg['AnoMes'], mensal_agg['Demanda'], alpha=0.8, color='teal', edgecolor='black')
    plt.title('Demanda Mensal ao Longo do Tempo')
    plt.xlabel('Ano-Mês')
    plt.ylabel('Demanda Total Mensal')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 6.3 Análise de sazonalidade mensal
    plt.subplot(4, 2, 4)
    meses = range(1, 13)
    demanda_mensal = [mensal_agg[mensal_agg['AnoMes'].str.endswith(f'-{m:02d}')]['Demanda'].mean() 
                     for m in meses]
    plt.bar(meses, demanda_mensal, alpha=0.7, color='coral', edgecolor='black')
    plt.title('Demanda Média por Mês do Ano (Sazonalidade)')
    plt.xlabel('Mês')
    plt.ylabel('Demanda Média')
    plt.xticks(meses)
    plt.grid(True, alpha=0.3)
    
    # 6.4 Top 10 dias de maior demanda
    plt.subplot(4, 2, 5)
    top_10_demanda = df.nlargest(10, 'Demanda')
    plt.bar(top_10_demanda.index.strftime('%Y-%m-%d'), top_10_demanda['Demanda'], color='purple', alpha=0.8)
    plt.title('Top 10 Dias com Maior Demanda')
    plt.xlabel('Data')
    plt.ylabel('Demanda')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6.5 Top 3 preços mais praticados
    plt.subplot(4, 2, 6)
    top_prices = df['Preco'].value_counts().nlargest(3)
    sns.barplot(x=top_prices.index, y=top_prices.values, palette='viridis', order=top_prices.index)
    plt.title('Top 3 Preços Mais Frequentes')
    plt.xlabel('Preço')
    plt.ylabel('Contagem de Dias')
    plt.grid(True, alpha=0.3)

    # 6.6 Autocorrelação da demanda
    plt.subplot(4, 2, 7)
    plot_acf(df['Log_Demanda'].dropna(), lags=30, ax=plt.gca())
    plt.title('Autocorrelação (ACF) - Log Demanda')
    plt.grid(True, alpha=0.3)
    
    # 6.7 Autocorrelação Parcial da demanda
    plt.subplot(4, 2, 8)
    plot_pacf(df['Log_Demanda'].dropna(), lags=30, ax=plt.gca())
    plt.title('Autocorrelação Parcial (PACF) - Log Demanda')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../Graficos/avaliacao_sku_{sku}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Análise de sazonalidade por dia da semana
    print("7. ANÁLISE DE SAZONALIDADE POR DIA DA SEMANA:")
    print("=" * 50)
    
    dias_semana = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 
                  'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
    
    demanda_por_dia = {}
    for dia in dias_semana:
        if dia in df.columns:
            demanda_por_dia[dia] = df[df[dia] == 1]['Demanda'].mean()
    
    if demanda_por_dia:
        print("Demanda média por dia da semana:")
        for dia in dias_semana:
            if dia in demanda_por_dia:
                print(f"  {dia}: {demanda_por_dia[dia]:.2f}")
    else:
        print("  Não foram encontradas colunas de dias da semana para análise.")
    
    # 8. Resumo da avaliação
    print("\n8. RESUMO E RECOMENDAÇÕES PARA MODELAGEM:")
    print("=" * 60)
    
    # Verificar estacionariedade
    nao_estacionarias = [col for col, est in resultados_estacionariedade.items() if not est]
    
    if nao_estacionarias:
        print(" VARIÁVEIS NÃO ESTACIONÁRIAS (precisam de transformação):")
        for var in nao_estacionarias:
            print(f"   - {var}")
        print("\n RECOMENDAÇÃO: Use diferenciação (df.diff()) ou transformações adicionais")
    else:
        print("TODAS as variáveis são estacionárias - OK para modelagem")
    
    # Verificar correlação
    correlacao = df['Log_Preco'].corr(df['Log_Demanda'])
    print(f"\n Correlação Log_Preco x Log_Demanda: {correlacao:.4f}")
    
    if abs(correlacao) < 0.2:
        print("   → Correlação MUITO fraca - preço pode não ser bom preditor")
    elif abs(correlacao) < 0.4:
        print("   → Correlação fraca a moderada")
    elif abs(correlacao) < 0.6:
        print("   → Correlação moderada - relação interessante")
    elif abs(correlacao) < 0.8:
        print("   → Correlação forte - bom preditor")
    else:
        print("   → Correlação MUITO forte - excelente preditor")
    
    # Recomendações finais
    print("\n RECOMENDAÇÕES FINAIS PARA VALIDAÇÃO CRUZADA:")
    print("=" * 60)
    
    if nao_estacionarias:
        print("1. TRANSFORME as variáveis não estacionárias antes da modelagem")
        print("2. Use diferenciação: df['var_diff'] = df['var'].diff().dropna()")
        print("3. Considere testar diferentes ordens de diferenciação")
    else:
        print("1. Variáveis já estacionárias - pode prosseguir diretamente")
    
    print("4. Use TimeSeriesSplit do sklearn para validação cruzada temporal")
    print("5. Monitore overfitting comparando performance treino/teste")
    print("6. Considere modelos SARIMA para capturar sazonalidade")
    
    print("\n Análise concluída - Dados prontos para modelagem!")
    
    return mensal_agg, resultados_estacionariedade

def plotar_comparacao_previsoes(df_previsoes, df_venda, sku):
    """
    Cria gráficos comparando a demanda real com as previsões dos modelos.

    Args:
        df_previsoes (pd.DataFrame): DataFrame com as previsões dos modelos.
        df_venda (pd.DataFrame): DataFrame com os dados históricos de vendas.
        sku (str): O SKU do produto para usar nos títulos e nome do arquivo.
    """
    print(f"--- GERANDO GRÁFICOS DE COMPARAÇÃO DE PREVISÕES PARA SKU {sku} ---")
    
    # Garantir que a coluna de data seja do tipo datetime em ambos os DataFrames
    df_previsoes['Data'] = pd.to_datetime(df_previsoes['Data'])
    
    # Definir a data como índice para facilitar a junção e plotagem
    df_previsoes_idx = df_previsoes.set_index('Data')
    
    # Juntar as previsões com os dados de vendas.
    # Usamos um 'left' join a partir das previsões para garantir que todas as datas de previsão sejam mantidas.
    # A demanda real (df_venda['Demanda']) será NaN para datas futuras onde não há histórico.
    df_comparacao = df_previsoes_idx.join(df_venda['Demanda'], how='left')
    
    if df_comparacao.empty:
        print("Não foi possível comparar os dados. Verifique se as datas nas previsões correspondem aos dados de vendas.")
        return

    # Configurar os gráficos
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
    
    # --- Gráfico 1: Demanda Real vs. Previsões ---
    axes[0].plot(df_comparacao.index, df_comparacao['Demanda'], label='Demanda Real', color='black', linewidth=2.5, marker='o', markersize=4, linestyle='--')
    axes[0].plot(df_comparacao.index, df_comparacao['previsao_SARIMAX'], label='Previsão SARIMAX', color='red', linewidth=2)
    axes[0].plot(df_comparacao.index, df_comparacao['previsao_TSCV'], label='Previsão TSCV', color='blue', linewidth=2)
    axes[0].set_title(f'Comparação: Demanda Real vs. Previsões - SKU {sku}')
    axes[0].set_ylabel('Demanda')
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # --- Gráfico 2: Erros (Resíduos) das Previsões ---
    erro_sarimax = df_comparacao['Demanda'] - df_comparacao['previsao_SARIMAX']
    erro_tscv = df_comparacao['Demanda'] - df_comparacao['previsao_TSCV']
    
    axes[1].plot(df_comparacao.index, erro_sarimax, label='Erro SARIMAX (Real - Previsto)', color='red', alpha=0.8)
    axes[1].plot(df_comparacao.index, erro_tscv, label='Erro TSCV (Real - Previsto)', color='blue', alpha=0.8)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1, label='Erro Zero')
    axes[1].set_title(f'Erro de Previsão (Resíduos) - SKU {sku}')
    axes[1].set_xlabel('Data')
    axes[1].set_ylabel('Erro de Demanda')
    axes[1].legend()
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'../Graficos/comparacao_previsoes_sku_{sku}.png', dpi=300, bbox_inches='tight')
    plt.show()