# CNN.py
#
# O código-fonte a seguir implementa um otimizador de inventário utilizando uma 
# abordagem de aprendizado profundo (deep learning) por meio de redes neurais 
# convolucionais (CNNs).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten
from tabulate import tabulate

#  Inicialmente, a classe InventoryOptimizer é definida, encapsulando todas as 
# funcionalidades do sistema

class InventoryOptimizer:
	
# A função __init__ da classe inicializa o objeto, recebendo o caminho do arqui-
# vo CSV contendo os dados do inventário como parâmetro.

    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = MinMaxScaler()
        self.model = self.build_model()
        
# O método load_data da classe é responsável por carregar os dados do arquivo 
# CSV especificado, utilizando a biblioteca pandas, e retornar um DataFrame. 

    def load_data(self):
        training_df = pd.read_csv(self.file_path)
        return training_df
        
# Em seguida, a função preprocess_data normaliza os dados, escalando as colunas 
# 'time_scale', 'filial_id' e 'quant_item' para o intervalo [0, 1] por meio da 
# classe MinMaxScaler do scikit-learn.

    def preprocess_data(self, df):
        df[['time_scale', 'filial_id', 'quant_item']] = self.scaler.fit_transform(
            df[['time_scale', 'filial_id', 'quant_item']])
        return df
        
# A função build_model constrói a arquitetura da rede neural convolucional (CNN) 
# utilizando a API do Keras. A CNN é composta por uma camada de convolução uni-
# dimensional, uma camada de achatamento (flatten), e duas camadas densas 
# (fully connected).

    def build_model(self):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(2, 1)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

# O método train_model realiza o treinamento da CNN, registrando as tendências 
# de Mean Squared Error (MSE) e perda (loss) ao longo das épocas.

    def train_model(self, X_train, y_train, X_test, y_test):
        mse_history = []
        loss_history = []

        for epoch in range(50):
            history = self.model.fit(X_train, y_train, epochs=1, batch_size=32,
                                     validation_data=(X_test, y_test), verbose=0)

            mse_history.append(history.history['val_loss'])
            loss_history.append(history.history['loss'])

        return mse_history, loss_history

# A função evaluate_model avalia o desempenho do modelo treinado sobre o conjun-
# to de teste, calculando e exibindo o MSE.

    def evaluate_model(self, X_test, y_test):
        loss = self.model.evaluate(X_test, y_test)
        print(f'Mean Squared Error on Test Set: {loss}')
        
# O método predict realiza predições utilizando a CNN sobre o conjunto de teste.

    def predict(self, X_test):
        return self.model.predict(X_test)
        
# A função inverse_transform efetua a inversão da escala de normalização, con-
# vertendo as predições para a escala original.

    def inverse_transform(self, data):
        data_min = self.scaler.data_min_[2]
        data_max = self.scaler.data_max_[2]
        return data * (data_max - data_min) + data_min

    def plot_actual_vs_predicted(self, results_df):
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Actual'], results_df['Predicted'])
        plt.xlabel('Actual Quantity of Items')
        plt.ylabel('Predicted Quantity of Items')
        plt.title('Actual vs Predicted Quantity of Items')
        plt.show()

    def plot_mse_trend(self, mse_history):
        plt.figure(figsize=(10, 6))
        plt.plot(mse_history, label='Validation MSE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.title('Overall Trend of MSE')
        plt.legend()
        plt.show()

    def plot_loss_trend(self, loss_history):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Function Trend')
        plt.legend()
        plt.show()

    def print_results_table(self, results_df):
        print(tabulate(results_df, headers='keys', tablefmt='pretty'))


def main():
    # Carregamos o CSV do inventário:
    file_path = '/Users/igormol/Downloads/vendas_20180102_20220826/inventory_07813893004.csv'
    inventory_optimizer = InventoryOptimizer(file_path)

    # Pre-processamento de dados:
    training_df = inventory_optimizer.load_data()
    training_df = inventory_optimizer.preprocess_data(training_df)

    # Treinamos a rede neural:
    X = training_df[['time_scale', 'filial_id']]
    y = training_df['quant_item']
    X = np.array(X).reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mse_history, loss_history = inventory_optimizer.train_model(X_train, y_train, X_test, y_test)
    inventory_optimizer.evaluate_model(X_test, y_test)

    predictions = inventory_optimizer.predict(X_test)
    predictions_original_scale = inventory_optimizer.inverse_transform(predictions.flatten())

    results_df = pd.DataFrame({'Actual': inventory_optimizer.inverse_transform(y_test.values),
                               'Predicted': predictions_original_scale})

# Os resultados são apresentados na forma de uma tabela utilizando a biblioteca 
# tabulate, seguida por gráficos que ilustram a relação entre as quantidades re-
# ais e preditas, a tendência geral do MSE e a evolução da função de perda du-
# rante o treinamento.

    inventory_optimizer.print_results_table(results_df)
    inventory_optimizer.plot_actual_vs_predicted(results_df)
    inventory_optimizer.plot_mse_trend(mse_history)
    inventory_optimizer.plot_loss_trend(loss_history)


if __name__ == "__main__":
    main()
