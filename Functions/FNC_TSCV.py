import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

def modelo_validacao_cruzada_series_temporais(df, sku,*X_cols, var_dpd,n_splits=10):
    """
    Modelo de validação cruzada para séries temporais com métricas completas
    """
    
    print(f"=== MODELO DE VALIDAÇÃO CRUZADA - SKU {sku} ===\n")
    
    #  VERIFICAR se há múltiplas variáveis
    
    X_cols = list(X_cols)
    
    # Se só tiver uma variável, garantir que seja 2D
    if len(X_cols) == 1:
        print("  Apenas uma variável preditora - ajustando dimensões...")
    
    y_col = var_dpd

    # VERIFICAR E LIMPAR DADOS ANTES DE PROCESSAR
    print("Verificando qualidade dos dados...")
    
    # Verificar valores infinitos ou muito grandes
    y = df[y_col].astype(float).values
    X = df[X_cols].astype(float)
    
    # Identificar problemas em y
    problemas_y = np.isinf(y) | np.isnan(y) | (np.abs(y) > 1e10)
    if problemas_y.any():
        print(f"Encontrados {problemas_y.sum()} valores problemáticos em {y_col}")
        print(f"Valores únicos problemáticos: {np.unique(y[problemas_y])}")
        
        # Remover ou corrigir linhas problemáticas
        indices_problema = np.where(problemas_y)[0]
        indices_validos = np.where(~problemas_y)[0]
        
        if len(indices_validos) > len(X_cols) * 2:  # Mínimo para modelagem
            print(f"   Removendo {len(indices_problema)} linhas problemáticas")
            y = y[indices_validos]
            X = X.iloc[indices_validos]
            dates = df.index[indices_validos]
        else:
            print(f"Dados insuficientes após limpeza. Usando fallback.")
            # Fallback: substituir por valores próximos
            from scipy import stats
            y[problemas_y] = np.nanmedian(y[~problemas_y])


    
    # Preparar dados 
    X = df[X_cols].astype(float)
    y = df[y_col].astype(float).values   
    dates = df.index
    
    #  GARANTIR que X seja 2D mesmo com uma variável
    if len(X_cols) == 1:
        X = X.values.reshape(-1, 1)  # Transforma 1D em 2D
    else:
        X = X.values
    
    # Configurar validação cruzada temporal
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Armazenar resultados
    resultados = {
        'rmse': [], 'wape': [], 'twape': [], 
        'coeficientes': [], 'interceptos': [], 'p_valores': [],
        'erro_medio': [], 'predictions': [], 'actuals': [],
        'aic': [], 'bic': []  
    }
    
    print(" Executando validação cruzada temporal...")
    print("=" * 60)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Separar dados
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        dates_test = dates[test_idx]
        
        #  Verificar dimensões
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)
        
        # Treinar modelo
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        # Previsões
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # WAPE (Weighted Absolute Percentage Error)
        wape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100
        
        # TWAPE (Time-Weighted Absolute Percentage Error)
        pesos_temporais = np.arange(1, len(y_test) + 1)
        twape = np.sum(np.abs(y_test - y_pred) * pesos_temporais) / np.sum(np.abs(y_test) * pesos_temporais) * 100
        
        # Erro médio
        erro_medio = np.mean(y_test - y_pred)
        
        # Teste de significância estatística 
        X_with_const = sm.add_constant(X_train.astype(float))  
        y_train_float = y_train.astype(float)  
        
        try:
            model_sm = sm.OLS(y_train_float, X_with_const).fit()
            p_values = model_sm.pvalues[1:] 
            aic = model_sm.aic
            bic = model_sm.bic
        except Exception as e:
            print(f"  Erro no teste de significância (fold {fold+1}): {e}")
            p_values = np.ones(len(X_cols))
            aic = np.nan
            bic = np.nan
        
        # Armazenar resultados
        resultados['rmse'].append(rmse)
        resultados['wape'].append(wape)
        resultados['twape'].append(twape)
        resultados['coeficientes'].append(model.coef_)
        resultados['interceptos'].append(model.intercept_)
        resultados['p_valores'].append(p_values)
        resultados['erro_medio'].append(erro_medio)
        resultados['predictions'].extend(y_pred)
        resultados['actuals'].extend(y_test)
        resultados['aic'].append(aic)
        resultados['bic'].append(bic)
        
        print(f" Fold {fold + 1}:")
        print(f"   Período teste: {dates_test[0].date()} a {dates_test[-1].date()}")
        print(f"   RMSE: {rmse:.4f}, WAPE: {wape:.2f}%")
    
    print("\n" + "=" * 60)
    print(" RESULTADOS FINAIS DO MODELO")
    print("=" * 60)
    
    # Métricas médias
    print(f" Métricas Médias nos {n_splits} folds:")
    print(f"   RMSE: {np.mean(resultados['rmse']):.4f} (±{np.std(resultados['rmse']):.4f})")
    print(f"   WAPE: {np.mean(resultados['wape']):.2f}% (±{np.std(resultados['wape']):.2f}%)")
    print(f"   TWAPE: {np.mean(resultados['twape']):.2f}% (±{np.std(resultados['twape']):.2f}%)")
    print(f"   AIC: {np.mean(resultados['aic']):.2f} (±{np.std(resultados['aic']):.2f})")
    print(f"   BIC: {np.mean(resultados['bic']):.2f} (±{np.std(resultados['bic']):.2f})")
    print(f"   Erro Médio: {np.mean(resultados['erro_medio']):.4f}")
    
    # Coeficientes médios
    coef_medio = np.mean(resultados['coeficientes'], axis=0)
    intercepto_medio = np.mean(resultados['interceptos'])
    p_valores_medio = np.mean(resultados['p_valores'], axis=0)
    
    print(f"\n Coeficientes do Modelo (média):")
    print(f"   Intercepto: {intercepto_medio:.6f}")
    
    #  Corrigir para caso de coeficiente único
    if len(X_cols) == 1:
        print(f"   {X_cols[0]}: {coef_medio:.6f} (p-valor: {p_valores_medio:.4f})")
    else:
        for i, col in enumerate(X_cols):
            significativo = "Ok! " if p_valores_medio[i] < 0.05 else "Atenção! "
            print(f"   {col}: {coef_medio[i]:.6f} {significativo}(p-valor: {p_valores_medio[i]:.4f})")
    
    # Análise de resíduos
    residuos = np.array(resultados['actuals']) - np.array(resultados['predictions'])
    
    print(f"\n Análise de Resíduos:")
    print(f"   Média dos resíduos: {np.mean(residuos):.6f}")
    print(f"   Std dos resíduos: {np.std(residuos):.6f}")
    print(f"   Resíduos dentro de ±2σ: {np.mean((residuos >= -2*np.std(residuos)) & (residuos <= 2*np.std(residuos))) * 100:.1f}%")
    
    # Significância geral do modelo
    try:
        X_full = sm.add_constant(X.astype(float))
        y_float = y.astype(float)
        model_full = sm.OLS(y_float, X_full).fit()
        
        print(f"\n Significância Estatística do Modelo (treinado com dados completos):")
        print(f"   F-statistic: {model_full.fvalue:.2f}")
        print(f"   Prob (F-statistic): {model_full.f_pvalue:.6f}")
        print(f"   AIC (dados completos): {model_full.aic:.2f}")
        print(f"   BIC (dados completos): {model_full.bic:.2f}")
    except Exception as e:
        print(f"  Erro na significância geral do modelo: {e}")
        model_full = None
    
    # Retornar resultados detalhados
    return {
        'metricas_medias': {
            'rmse': np.mean(resultados['rmse']),
            'wape': np.mean(resultados['wape']),
            'twape': np.mean(resultados['twape']),
            'aic': np.mean(resultados['aic']),
            'bic': np.mean(resultados['bic']),
            'erro_medio': np.mean(resultados['erro_medio'])
        },
        'coeficientes': coef_medio,
        'intercepto': intercepto_medio,
        'p_valores': p_valores_medio,
        'residuos': residuos,
        'modelo_completo': model_full,
        'resultados_detalhados': resultados
    }

def converter_para_escala_original(resultados, df):
    """
    Converte previsões para escala original e calcula métricas
    """
    log_predictions = np.array(resultados['resultados_detalhados']['predictions'])
    log_actuals = np.array(resultados['resultados_detalhados']['actuals'])
    
    # Converter para escala original
    predictions_orig = np.exp(log_predictions)
    actuals_orig = np.exp(log_actuals)
    
    # Calcular métricas em escala original
    rmse_orig = np.sqrt(mean_squared_error(actuals_orig, predictions_orig))
    wape_orig = np.sum(np.abs(actuals_orig - predictions_orig)) / np.sum(actuals_orig) * 100
    
    print(f"\n Métricas em Escala Original:")
    print(f"   RMSE: {rmse_orig:.2f}")
    print(f"   WAPE: {wape_orig:.2f}%")
    
    return {
        'rmse_original': rmse_orig,
        'wape_original': wape_orig,
        'predictions_original': predictions_orig,
        'actuals_original': actuals_orig
    }

# Hyperparameter Tuning para buscar o melhor alpha

def encontrar_melhor_alpha(df, X_cols, y_col, alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]):
    """
    Encontra o melhor alpha para Ridge Regression usando validação cruzada temporal
    """
    X = df[X_cols].astype(float).values
    y = df[y_col].astype(float).values
    
    if len(X_cols) == 1:
        X = X.reshape(-1, 1)
    
    tscv = TimeSeriesSplit(n_splits=5)
    resultados_alpha = {}
    
    for alpha in alphas:
        rmse_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
                X_test = X_test.reshape(-1, 1)
            
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse)
        
        resultados_alpha[alpha] = np.mean(rmse_scores)
        print(f"Alpha {alpha}: RMSE médio = {resultados_alpha[alpha]:.4f}")
    
    # Encontrar melhor alpha
    melhor_alpha = min(resultados_alpha, key=resultados_alpha.get)
    print(f"\n MELHOR ALPHA: {melhor_alpha} (RMSE: {resultados_alpha[melhor_alpha]:.4f})")
    
    return melhor_alpha, resultados_alpha