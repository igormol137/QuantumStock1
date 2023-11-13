# Hybrid_CNN_RNN.py
#
# O código em questão implementa um otimizador de inventário utilizando uma 
# abordagem híbrida de redes neurais convolucionais (CNN) e recorrentes (RNN). 
# Este sistema é encapsulado na classe OtimizadorInventario, que é projetada 
# para facilitar a modularidade e a compreensão do código.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten
import matplotlib.pyplot as plt
from tabulate import tabulate

class InventoryOptimizer:
	
# A primeira função, __init__, inicializa a classe com informações cruciais, 
# como o caminho do arquivo de dados, o número de passos temporais e o número de 
# recursos. Esses parâmetros fornecem a flexibilidade necessária para ajustar o 
# comportamento do otimizador.
	
    def __init__(self, file_path, time_steps=10, n_features=3):
        self.file_path = file_path
        self.time_steps = time_steps
        self.n_features = n_features
        self.training_df = None
        self.scaler = MinMaxScaler()
        self.model = self.build_model()
        
# O método carregar_dados é responsável por carregar os dados do arquivo especi-
# ficado pelo caminho fornecido na inicialização. Esta função utiliza a biblio-
# teca pandas para ler os dados tabulares contidos no arquivo CSV.

    def load_data(self):
        self.training_df = pd.read_csv(self.file_path)

# A normalização dos dados é realizada pela função normalizar_dados, que utiliza 
# a classe MinMaxScaler da biblioteca scikit-learn para dimensionar os atributos 
# do conjunto de dados para o intervalo [0, 1]. Essa etapa é crucial para melho-
# rar a convergência e o desempenho do modelo durante o treinamento.

    def normalize_data(self):
        self.training_df['time_scale'] = self.scaler.fit_transform(self.training_df['time_scale'].values.reshape(-1, 1))
        self.training_df['filial_id'] = self.scaler.fit_transform(self.training_df['filial_id'].values.reshape(-1, 1))
        self.training_df['quant_item'] = self.scaler.fit_transform(self.training_df['quant_item'].values.reshape(-1, 1))

# A função criar_sequencias é empregada para gerar sequências temporais apropri-
# adas a partir dos dados. Nela, cada sequência é formada por uma janela desli-
# zante de passos temporais, com os rótulos representando a quantidade de itens 
# na posição temporal subsequente.

    def create_sequences(self, data):
        sequences = []
        labels = []
        for i in range(len(data) - self.time_steps):
            seq = data[i:i+self.time_steps]
            label = data[i+self.time_steps, -1]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

# A preparação das sequências para treinamento é realizada pela seguinte função, 
# que extrai as colunas relevantes do DataFrame e ajusta as dimensões das se-
# quências de acordo com as expectativas do modelo híbrido CNN-RNN.

    def prepare_sequences(self):
        X = self.training_df[['time_scale', 'filial_id', 'quant_item']].values
        y = self.training_df['quant_item'].values
        sequences, labels = self.create_sequences(X)
        sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], self.n_features))
        return sequences, labels

# A função construir_modelo define a arquitetura do modelo híbrido CNN-RNN uti-
# lizando a biblioteca TensorFlow e Keras. A arquitetura consiste em uma camada 
# convolucional 1D seguida por uma camada de pooling, uma camada LSTM e, final-
# mente, uma camada densa de saída.

    def build_model(self):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.time_steps, self.n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
        
# O treinamento do modelo é executado pela função treinar_modelo, que utiliza o 
# otimizador Adam e a métrica de erro quadrático médio (MSE) para ajustar os 
# pesos do modelo às sequências de treinamento.
# fazer_predicoes gera previsões utilizando o modelo treinado nas sequências de 
# treinamento. Essas previsões são posteriormente utilizadas para avaliar o 
# desempenho do modelo por meio de métricas como MSE, MAE, R2 Score e Residuais, 
# implementadas na função calcular_metricas.

    def train_model(self, sequences, labels, epochs=50):
        history = self.model.fit(sequences, labels, epochs=epochs, verbose=1)
        return history

    def make_predictions(self, sequences):
        return self.model.predict(sequences)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data).flatten()

    def calculate_metrics(self, labels, predictions):
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        residuals = labels - predictions.flatten()
        return mse, mae, r2, residuals

    def create_results_df(self, labels, predictions, mse, mae, r2, residuals):
        results_df = pd.DataFrame({
            'Actual': self.inverse_transform(labels.reshape(-1, 1)).flatten(),
            'Predicted': self.inverse_transform(predictions).flatten(),
            'MSE': [mse] * len(labels),
            'MAE': [mae] * len(labels),
            'R2 Score': [r2] * len(labels),
            'Residuals': residuals
        })
        return results_df
        
# Para exibição e análise mais detalhada, a função imprimir_tabela_resultados 
# utiliza a biblioteca tabulate para apresentar os resultados de forma estrutu-
# rada e legível em formato de tabela. Por fim, as funções 
# plotar_grafico_real_vs_previsto, plotar_tendencia_mse e plotar_tendencia_perda 
# geram gráficos visuais para auxiliar na interpretação dos resultados, exibindo 
# as relações entre os valores reais e previstos, a tendência do MSE ao longo das 
# épocas de treinamento e a evolução da função de perda, respectivamente.

    def print_results_table(self, results_df):
        print(tabulate(results_df, headers='keys', tablefmt='fancy_grid'))

    def plot_actual_vs_predicted(self, results_df):
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['Actual'], label='Actual', marker='o')
        plt.plot(results_df['Predicted'], label='Predicted', marker='o')
        plt.title('Actual vs Predicted Quantity of Items')
        plt.xlabel('Data Point')
        plt.ylabel('Quantity of Items (Scaled)')
        plt.legend()
        plt.show()

    def plot_mse_trend(self, history):
        plt.figure(figsize=(10, 6))
        mse_key = [key for key in history.history.keys() if 'mean_squared_error' in key.lower()]
        mse_key = mse_key[0] if mse_key else 'loss'
        plt.plot(history.history[mse_key], label='MSE')
        plt.title('Overall Trend of Mean Squared Error (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

    def plot_loss_trend(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Loss')
        plt.title('Loss Function Trend')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

#     A função "main()" do código é evocada a seguir, responsável por orquestrar 
# o processo de otimização de inventário utilizando um modelo híbrido de redes 
# neurais convolucionais (CNN) e recorrentes (RNN). A primeira ação realizada é 
# a especificação do caminho do arquivo contendo os dados de inventário em for-
# mato CSV.
#     Em seguida, um objeto da classe OtimizadorInventario é instanciado, rece-
# bendo o caminho do arquivo como parâmetro de inicialização. Este objeto é 
# essencial para a execução das operações subsequentes.
#     O método carregar_dados é invocado para carregar os dados do inventário a 
# partir do arquivo especificado. Posteriormente, os dados são normalizados 
# através da chamada do método normalizar_dados, que utiliza a técnica de escala 
# "min-max" para ajustar os valores dos atributos no intervalo [0, 1].
#     A função preparar_sequencias é então acionada para gerar sequências tempo-
# rais e rótulos apropriados a partir dos dados normalizados. Essas sequências e 
# rótulos são fundamentais para o treinamento subsequente do modelo. O método 
# treinar_modelo é utilizado para efetuar o treinamento do modelo híbrido 
# CNN-RNN, utilizando as sequências e rótulos preparados anteriormente. O his-
# tórico do treinamento é armazenado para análise posterior.
#     Com o modelo treinado, a função fazer_predicoes é invocada para gerar pre-
# visões utilizando as sequências de treinamento. As métricas de desempenho, 
# como erro quadrático médio (MSE), erro absoluto médio (MAE), coeficiente de 
# determinação (R2) e resíduos, são então calculadas através da chamada de 
# calcular_metricas.
#     O resultado final é consolidado em um DataFrame denominado df_resultados 
# utilizando o método criar_df_resultados. Esse DataFrame encapsula as previsões, 
# os rótulos reais e as métricas de desempenho, proporcionando uma visão abran-
# gente dos resultados obtidos no processo de otimização de inventário.

def main():
    file_path = '/Users/igormol/Downloads/vendas_20180102_20220826/inventory_07813893004.csv'
    optimizer = InventoryOptimizer(file_path)
    optimizer.load_data()
    optimizer.normalize_data()
    sequences, labels = optimizer.prepare_sequences()
    history = optimizer.train_model(sequences, labels)
    predictions = optimizer.make_predictions(sequences)
    mse, mae, r2, residuals = optimizer.calculate_metrics(labels, predictions)
    results_df = optimizer.create_results_df(labels, predictions, mse, mae, r2, residuals)
    optimizer.plot_actual_vs_predicted(results_df)
    optimizer.plot_mse_trend(history)
    optimizer.plot_loss_trend(history)
    optimizer.print_results_table(results_df)

if __name__ == "__main__":
    main()
