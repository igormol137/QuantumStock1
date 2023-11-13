# GRU.py
#
# O código apresentado refere-se a uma implementação de otimização de inventário 
# utilizando Redes Neurais Recorrentes (RNNs) com uma arquitetura de Unidade 
# Recorrente Gated (GRU). 

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# A classe InventoryOptimizer é inicializada com os parâmetros essenciais, como 
# o caminho do arquivo CSV contendo os dados do inventário, o número de passos 
# temporais (time_steps), as unidades GRU, o número de épocas e o tamanho do 
# lote para o treinamento da rede neural.

class InventoryOptimizer:
		
    def __init__(self, csv_file_path, time_steps=10, gru_units=50, epochs=50, batch_size=32):
        self.csv_file_path = csv_file_path
        self.time_steps = time_steps
        self.gru_units = gru_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = self.build_model()

# O método load_data realiza a leitura do arquivo CSV e normaliza os dados usan-
# do o MinMaxScaler da biblioteca scikit-learn. Os dados normalizados são então
# divididos em sequências utilizando o método create_sequences. Esta etapa é 
# crucial para a preparação dos dados de entrada para a rede neural, onde cada 
# sequência representa uma observação temporal do inventário.	

    def load_data(self):
        training_df = pd.read_csv(self.csv_file_path)
        scaled_data = self.scaler.fit_transform(training_df[['time_scale', 'filial_id', 'quant_item']])
        return scaled_data, training_df

    def create_sequences(self, data):
        sequences = []
        target = []
        for i in range(len(data) - self.time_steps):
            seq = data[i : (i + self.time_steps), :]
            label = data[i + self.time_steps, 2]  # Assuming 'quant_item' is the target variable
            sequences.append(seq)
            target.append(label)
        return np.array(sequences), np.array(target)

# O método build_model cria uma arquitetura de rede neural sequencial, composta 
# por uma camada GRU e uma camada densa. A rede é compilada com a função de per-
# da Mean Squared Error (MSE) e o otimizador Adam. O treinamento do modelo é 
# executado pelo método train_model, que utiliza os dados de treinamento e vali-
# dação, retornando o histórico de treinamento. Essa abordagem permite a análise 
# do desempenho do modelo ao longo das épocas.

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.GRU(self.gru_units, activation='relu', input_shape=(self.time_steps, 3)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model(self, X_train, y_train, X_test, y_test):
        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_test, y_test))
        return history

# Após o treinamento, o método predict é empregado para realizar previsões sobre 
# os dados de teste. A inversão da transformação normalizada é realizada com o 
# método inverse_transform, permitindo a comparação dos valores previstos com os 
# valores reais. A classe também oferece métodos para imprimir os resultados na 
# forma de uma tabela, salvar os resultados em um arquivo CSV e visualizar os 
# resultados por meio de gráficos, fornecendo uma análise abrangente e detalhada 
# da otimização do inventário.

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

    def inverse_transform(self, data):
        return data * (self.scaler.data_max_[2] - self.scaler.data_min_[2]) + self.scaler.data_min_[2]

    def print_results(self, results_df):
        print(results_df)

    def save_results_to_csv(self, results_df, filename='results.csv'):
        results_df.to_csv(filename, index=False)

    def plot_results(self, results_df, history):
        # Traça o gráfico do inventário ótimo versus os dados reais:
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['Time'], results_df['Actual_Quantity'], label='Actual Quantity')
        plt.plot(results_df['Time'], results_df['Predicted_Quantity'], label='Predicted Quantity')
        plt.xlabel('Time')
        plt.ylabel('Quantity')
        plt.title('Optimal Quantity vs Actual Data')
        plt.legend()
        plt.show()

        # Traça a função perda:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Function during Training')
        plt.legend()
        plt.show()

def main():
    optimizer = InventoryOptimizer(csv_file_path='/Users/igormol/Downloads/vendas_20180102_20220826/inventory_07813893004.csv')
    scaled_data, training_df = optimizer.load_data()
    X, y = optimizer.create_sequences(scaled_data)
    
    # Divide os dados entre conjuntos de treino e teste, respectivamente:
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    history = optimizer.train_model(X_train, y_train, X_test, y_test)
    predictions = optimizer.predict(X_test)

    # Pré-processamento dos resultados:
    predicted_values = optimizer.inverse_transform(predictions)
    actual_values = optimizer.inverse_transform(y_test)

    # Organiza os resultados em uma estrutura de dados:
    results_df = pd.DataFrame({
        'Time': training_df['time_scale'].iloc[split + optimizer.time_steps:].reset_index(drop=True),
        'Filial_ID': training_df['filial_id'].iloc[split + optimizer.time_steps:].reset_index(drop=True),
        'Actual_Quantity': actual_values.flatten(),
        'Predicted_Quantity': predicted_values.flatten()
    })

    optimizer.print_results(results_df)
    optimizer.save_results_to_csv(results_df)
    optimizer.plot_results(results_df, history)

if __name__ == "__main__":
    main()
