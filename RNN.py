# RNN.py
#
# O código-fonte a seguir implementa um modelo de aprendizado profundo para oti-
# mização de inventário usando uma Rede Neural Recorrente (RNN) com TensorFlow e 
# Keras. A estrutura foi organizada em uma classe chamada InventoryOptimizer pa-
# ra modularidade e clareza.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

class InventoryOptimizer:
	
# A função __init__(self, data) inicializa a classe InventoryOptimizer com o 
# conjunto de dados fornecido. O construtor também configura parâmetros essenci-
# ais, como o comprimento da sequência (sequence_length), o objeto MinMaxScaler 
# para normalização dos dados e o modelo RNN.
	
    def __init__(self, data):
        self.data = data
        self.sequence_length = 10
        self.scaler = MinMaxScaler()
        self.model = self.build_model()
        
# A função preprocess_data(self) realiza a normalização dos dados e cria sequên-
# cias para treinamento da RNN. Os dados são escalados para o intervalo [0, 1] 
# usando o MinMaxScaler. As sequências de entrada (sequences) e os rótulos 
# correspondentes (target) são gerados a partir dos dados normalizados.

    def preprocess_data(self):
        scaled_data = self.scaler.fit_transform(self.data[['time_scale', 'filial_id', 'quant_item']])
        sequences, target = [], []

        for i in range(len(scaled_data) - self.sequence_length):
            sequences.append(scaled_data[i:i+self.sequence_length])
            target.append(scaled_data[i+self.sequence_length, 2])

        sequences, target = np.array(sequences), np.array(target)
        return sequences, target
        
# A função build_model(self) constrói o modelo RNN usando a biblioteca Keras.
# O modelo consiste em uma camada RNN com ativação ReLU e uma camada densa de 
# saída. A função de perda utilizada é o erro quadrático médio (MSE), e o otimi-
# zador é o Adam.

    def build_model(self):
        model = Sequential()
        model.add(SimpleRNN(50, activation='relu', input_shape=(self.sequence_length, 3)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
        
# A função train_model treina o modelo RNN com os dados de treinamento. Os parâ-
# metros padrão são configurados para 50 épocas e um tamanho de lote de 32. 
# A função retorna o histórico de treinamento, incluindo as métricas de MSE para 
# o conjunto de treinamento e validação.

    def train_model(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        return history
        
# A função evaluate_model(self, X_test, y_test) avalia o desempenho do modelo 
# nos dados de teste e imprime o MSE resultante. Isso fornece uma medida quanti-
# tativa da qualidade do modelo em dados não vistos.

    def evaluate_model(self, X_test, y_test):
        loss = self.model.evaluate(X_test, y_test)
        print(f'Mean Squared Error on Test Data: {loss}')

# A função predict(self, X_test) gera previsões para os dados de teste usando o 
# modelo treinado. As previsões são então desnormalizadas para a escala original 
# dos dados.

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return self.scaler.inverse_transform(np.concatenate([X_test[:, -1, :2], predictions.reshape(-1, 1)], axis=1))[:, -1]

# A função print_results_table imprime os resultados da otimização de inventório
# em uma tabela.

    def print_results_table(self, actual, predicted):
        results_df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
        print(results_df)
        
# As funções plot_results, plot_mse_trend e plot_loss_function são responsáveis 
# por gerar gráficos que visualizam os resultados do modelo. A primeira gera um 
# gráfico de dispersão comparando os valores reais e previstos, enquanto as duas 
# últimas geram gráficos que mostram a tendência geral do MSE ao longo do trei-
# namento e a função de perda durante o treinamento, respectivamente.

    def plot_results(self, actual, predicted):
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted)
        plt.xlabel('Actual Number of Items')
        plt.ylabel('Predicted Number of Items')
        plt.title('Optimal Number of Items vs Actual Number of Items')
        plt.show()

    def plot_mse_trend(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['val_loss'], label='Validation MSE')
        plt.plot(history.history['loss'], label='Training MSE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.title('Overall Trend of Mean Squared Error (MSE)')
        plt.legend()
        plt.show()

    def plot_loss_function(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Function During Training')
        plt.legend()
        plt.show()

# Por fim, a função main() instancia a classe InventoryOptimizer, realiza a pre-
# paração dos dados, treina o modelo, avalia seu desempenho, gera previsões e 
# cria visualizações dos resultados. Este padrão de execução ajuda a manter o 
# código organizado e facilita a reutilização em diferentes cenários de otimiza-
# ção de inventário.

def main():
    # Assume training_df is your DataFrame
    # For simplicity, I'll create a sample DataFrame
    # Replace this with your actual data
    data = {'time_scale': range(1, 100),
            'filial_id': np.random.randint(1, 4, size=99),
            'quant_item': np.random.randint(10, 100, size=99)}

    training_df = pd.DataFrame(data)

    inventory_optimizer = InventoryOptimizer(training_df)

    # Pré-processamento de dados:
    X, y = inventory_optimizer.preprocess_data()

    # Divisão de dados em conjuntos de treino e teste, respectivamente:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinamento do modelo:
    history = inventory_optimizer.train_model(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    inventory_optimizer.evaluate_model(X_test, y_test)

    # Gera previsões:
    predictions = inventory_optimizer.predict(X_test)

    # Imprime resultados em uma tabela:
    inventory_optimizer.print_results_table(actual=inventory_optimizer.scaler.inverse_transform(np.concatenate([X_test[:, -1, :2], y_test.reshape(-1, 1)], axis=1))[:, -1], predicted=predictions)

    # Traça os gráficos do modelo:
    inventory_optimizer.plot_results(inventory_optimizer.scaler.inverse_transform(np.concatenate([X_test[:, -1, :2], y_test.reshape(-1, 1)], axis=1))[:, -1], predictions)
    inventory_optimizer.plot_mse_trend(history)
    inventory_optimizer.plot_loss_function(history)

if __name__ == "__main__":
    main()
