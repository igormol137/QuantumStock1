# Hybrid_CNN_LSTM.py
#
# Este programa em Python implementa uma abordagem de aprendizado profundo hí-
# brido utilizando redes neurais convolucionais (CNN) e redes neurais do tipo
# Long Short-Term Memory (LSTM) para resolver um problema de otimização de in-
#ventário.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Reshape
from tabulate import tabulate
import matplotlib.pyplot as plt

# Define a classe InventoryOptimizer para encapsular as funcionalidades do pro-
# grama. O construtor (__init__) inicializa as variáveis de caminho do arquivo, 
# conjuntos de dados, modelo e objetos de escalonamento.

class InventoryOptimizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.training_df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.scaler_X, self.scaler_y = MinMaxScaler(), MinMaxScaler()

# Carrega os dados do arquivo CSV especificado (file_path) em um DataFrame do 
# Pandas e os ordena com base na coluna 'time_scale'.

    def load_data(self):
        self.training_df = pd.read_csv(self.file_path)
        self.training_df.sort_values(by='time_scale', inplace=True)

# Extrai as features (X) e os rótulos (y) do DataFrame. Normaliza as features e 
# divide os dados em conjuntos de treinamento e teste usando o escalonador 
# MinMaxScaler.

    def preprocess_data(self):
        X = self.training_df[['time_scale', 'filial_id']].values
        y = self.training_df['quant_item'].values

        # Normalização:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.scaler_X.fit_transform(X).reshape((X.shape[0], X.shape[1], 1)),
            self.scaler_y.fit_transform(y.reshape(-1, 1)),
            test_size=0.2,
            random_state=42
        )

# Constrói o modelo de rede neural sequencial com uma camada Conv1D, uma camada 
# Flatten, uma camada Reshape e uma camada LSTM. Compila o modelo com o otimi-
# zador 'adam' e a função de perda 'mse' (Mean Squared Error).

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(self.X_train.shape[1], 1)))
        self.model.add(Flatten())
        self.model.add(Reshape((self.X_train.shape[1], -1)))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

# Treina o modelo utilizando os dados de treinamento, especificando o número de 
# épocas. Retorna o histórico do treinamento para análise posterior.

    def train_model(self, num_epochs):
        history = self.model.fit(self.X_train, self.y_train, epochs=num_epochs, batch_size=16, validation_data=(self.X_test, self.y_test), verbose=0)
        return history

# Gera previsões do modelo para os dados de teste, realiza a inversão da escala 
# para obter valores reais e retorna as previsões como um array unidimensional.

    def predict(self):
        y_pred_scaled = self.model.predict(self.X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred.flatten()

# Calcula métricas como desvio padrão, erro estatístico e resíduos com base nas 
# previsões e nos valores reais.

    def calculate_metrics(self, y_pred):
        results_df = pd.DataFrame({'Actual': self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).flatten(),
                                   'Predicted': y_pred})

        std_dev = np.std(results_df['Actual'] - results_df['Predicted'])
        stat_error = np.abs(results_df['Actual'] - results_df['Predicted']) / results_df['Actual']
        residuals = results_df['Actual'] - results_df['Predicted']

        results_df['Standard Deviation'] = std_dev
        results_df['Statistical Error'] = stat_error
        results_df['Residuals'] = residuals

        return results_df
        
# Imprime uma tabela formatada dos resultados usando a biblioteca Tabulate.

    def print_results_table(self, results_df):
        print(tabulate(results_df, headers='keys', tablefmt='fancy_grid'))

# Gera um gráfico de dispersão comparando os valores reais e previstos.

    def plot_actual_vs_predicted(self, results_df):
        plt.figure(figsize=(10, 5))
        plt.scatter(results_df['Actual'], results_df['Predicted'])
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()
        
# Gera um gráfico que mostra a tendência do erro médio quadrático (MSE) ao longo 
# das épocas durante o treinamento.

    def plot_mse_trend(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training MSE')
        plt.plot(history.history['val_loss'], label='Validation MSE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.title('Overall Trend of Mean Squared Error')
        plt.legend()
        plt.show()
        
# Função principal que instancia a classe InventoryOptimizer, carrega dados, 
# pré-processa, constrói, treina o modelo, faz previsões, calcula métricas e 
# exibe resultados.

def main():
    file_path = "/Users/igormol/Downloads/vendas_20180102_20220826/inventory_07813893004.csv"
    inventory_optimizer = InventoryOptimizer(file_path)
    inventory_optimizer.load_data()
    inventory_optimizer.preprocess_data()
    inventory_optimizer.build_model()
    
    num_epochs = 50

    history = inventory_optimizer.train_model(num_epochs)
    y_pred = inventory_optimizer.predict()
    results_df = inventory_optimizer.calculate_metrics(y_pred)

    inventory_optimizer.print_results_table(results_df)
    inventory_optimizer.plot_actual_vs_predicted(results_df)
    inventory_optimizer.plot_mse_trend(history)

if __name__ == "__main__":
    main()
