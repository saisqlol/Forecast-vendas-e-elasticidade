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
    
    numeric_cols = ['Preco', 'Demanda', 'Log_Preco','Log_Preco_Listado', 'Log_Demanda', 
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
    fig = plt.figure(figsize=(20, 20))
    
    # 6.1 Série temporal original com média móvel
    plt.subplot(3, 2, 1)
    plt.plot(df.index, df['Preco'], 'b-', label='Preço', alpha=0.5)
    plt.plot(df.index, df['Preco'].rolling(7).mean(), 'darkblue', label='MM7 Preço', linewidth=2)
    plt.title('Série Temporal - Preço (com média móvel 7 dias)')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    plt.plot(df.index, df['Demanda'], 'r-', label='Demanda', alpha=0.5)
    plt.plot(df.index, df['Demanda'].rolling(7).mean(), 'darkred', label='MM7 Demanda', linewidth=2)
    plt.title('Série Temporal - Demanda (com média móvel 7 dias)')
    plt.xlabel('Data')
    plt.ylabel('Demanda')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6.2 Análise de sazonalidade mensal
    plt.subplot(3, 2, 3)
    meses = range(1, 13)
    demanda_mensal = [mensal_agg[mensal_agg['AnoMes'].str.endswith(f'-{m:02d}')]['Demanda'].mean() 
                     for m in meses]
    plt.bar(meses, demanda_mensal, alpha=0.7, color='coral', edgecolor='black')
    plt.title('Demanda Média por Mês (Sazonalidade)')
    plt.xlabel('Mês')
    plt.ylabel('Demanda Média')
    plt.grid(True, alpha=0.3)
    
    # 6.3 Scatter plot - Relação Preço x Demanda
    plt.subplot(3, 2, 4)
    plt.scatter(df['Log_Preco'], df['Log_Demanda'], alpha=0.6, color='green')
    plt.xlabel('Log Preço')
    plt.ylabel('Log Demanda')
    plt.title('Relação Log Preço vs Log Demanda')
    plt.grid(True, alpha=0.3)
    
    # 6.4 Autocorrelação da demanda
    plt.subplot(3, 2, 5)
    plot_acf(df['Log_Demanda'].dropna(), lags=30, ax=plt.gca())
    plt.title('Autocorrelação - Log Demanda')
    plt.grid(True, alpha=0.3)
    
    # 6.5 Distribuição dos dados
    plt.subplot(3, 2, 6)
    plt.hist(df['Log_Demanda'].dropna(), bins=30, alpha=0.7, color='lightblue', 
             edgecolor='black', density=True, label='Log Demanda')
    plt.hist(df['Log_Preco'].dropna(), bins=30, alpha=0.7, color='lightgreen', 
             edgecolor='black', density=True, label='Log Preço')
    plt.title('Distribuição das Variáveis em Escala Log')
    plt.xlabel('Valor em Log')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'avaliacao_sku_{sku}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Análise de sazonalidade por dia da semana
    print("7. ANÁLISE DE SAZONALIDADE POR DIA DA SEMANA:")
    print("=" * 50)
    
    dias_semana = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 
                  'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
    
    demanda_por_dia = {}
    for i, dia in enumerate(dias_semana, 1):
        if f'Dia_{i}' in df.columns:
            demanda_por_dia[dia] = df[df[f'Dia_{i}'] == 1]['Demanda'].mean()
    
    if demanda_por_dia:
        print("Demanda média por dia da semana:")
        for dia, media in demanda_por_dia.items():
            print(f"  {dia}: {media:.2f}")
    
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