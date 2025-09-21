import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PredictorVendas:
    def __init__(self, resultados_modelo):
        """
        Inicializa o predictor com os resultados do modelo treinado
        """
        self.intercepto = resultados_modelo['intercepto']
        self.coeficientes = resultados_modelo['coeficientes']
        
        if 'feature_names' in resultados_modelo:
            self.feature_names = resultados_modelo['feature_names']
        else:
            self.feature_names = ['Log_Preco', 'Black_Friday', 'promocionado_25', 
                                'Quarta-feira', 'Terça-feira']
        
        print("Predictor inicializado com sucesso!")
        print(f"   Intercepto: {self.intercepto:.6f}")
        for i, feature in enumerate(self.feature_names):
            print(f"   {feature}: {self.coeficientes[i]:.6f}")
    
    def preparar_features(self, df):
        """
        Prepara as features para previsão a partir dos dados brutos
        """
        df = df.copy()
        
        # Garantir que a data seja datetime
        if 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'])
        
        # Calcular Log_Preco
        df['Log_Preco'] = np.log(df['Preco'])
        
        # Criar dummies para dias da semana
        dias_map = {
            0: 'Segunda-feira',
            1: 'Terça-feira', 
            2: 'Quarta-feira',
            3: 'Quinta-feira',
            4: 'Sexta-feira',
            5: 'Sábado',
            6: 'Domingo'
        }
        
        # Adicionar colunas de dias da semana
        for dia_num, dia_nome in dias_map.items():
            df[dia_nome] = (df['Data'].dt.dayofweek == dia_num).astype(int)
        
        return df
    
    def prever_demanda(self, df_input):
        """
        Faz previsões de demanda para os dados fornecidos
        
        Parameters:
        df_input: DataFrame com colunas 'Data', 'SKU', 'Preco'
        
        Returns:
        DataFrame com previsões adicionadas
        """
        # Preparar features
        df = self.preparar_features(df_input)
        
        # Garantir que todas as features necessárias existam
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Features faltando: {missing_features}")
        
        # Selecionar apenas as features do modelo
        X = df[self.feature_names]
        
        # Fazer previsões em escala log
        log_demanda_prevista = self.intercepto + X.dot(self.coeficientes)
        
        # Converter para escala original
        demanda_prevista = np.exp(log_demanda_prevista)
        
        # Adicionar previsões ao DataFrame original
        resultado = df_input.copy()
        resultado['Log_Demanda_Prevista'] = log_demanda_prevista.values
        resultado['Demanda_Prevista'] = demanda_prevista.values
        
        return resultado

def criar_predictor(resultados_modelo):
    """
    Função conveniente para criar o predictor
    
    Parameters:
    resultados_modelo: dict do modelo treinado
    
    Returns:
    PredictorVendas instance
    """
    return PredictorVendas(resultados_modelo)

def prever_planilha(resultados_modelo, caminho_planilha):
    """
    Função completa para prever a partir de uma planilha Excel
    
    Parameters:
    resultados_modelo: dict do modelo treinado
    caminho_planilha: caminho para o arquivo Excel
    
    Returns:
    DataFrame com previsões
    """
    # Criar predictor
    predictor = criar_predictor(resultados_modelo)
    
    # Ler planilha
    df_input = pd.read_excel(caminho_planilha)
    
    # Fazer previsões
    return predictor.prever_demanda(df_input)
