# Anomaly_Detection.py
#
# O código-fonte a seguir apresenta uma implementação de uma abordagem híbrida 
# para a análise de inventário, especificamente, a detecção e substituição de
# anomalias, empregando Redes Neurais Convolucionais (CNN) e Isolation Forest.

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tabulate import tabulate

class InventoryAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.training_df = None
        self.results_df = None
        self.clean_df = None

# Carregamento de Dados:
#    - load_data(): Inicia o processo carregando dados brutos de inventário de 
# um arquivo CSV.

    def load_data(self):
        self.training_df = pd.read_csv(self.file_path)

# Pré-processamento de Dados:
#    - scale_data(): Padroniza os dados de quantidade do item usando Standard-
# Scaler do scikit-learn.

    def scale_data(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.training_df[['quant_item']])
        return scaled_data

# Construção e Treinamento do Modelo CNN:
#    - build_cnn_model(): Configura arquitetura CNN usando Keras/TensorFlow.
#    - train_cnn_model(): Realiza treinamento do CNN com dados padronizados.

    def build_cnn_model(self, input_shape):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
        model.add(Flatten())
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=1, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_cnn_model(self, model, X_train, y_train, num_epochs=20):
        return model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, verbose=2)

# Predição e Cálculo do Erro de Reconstrução:
#    - calculate_reconstruction_error(): Calcula o erro de reconstrução entre 
# previsões do modelo CNN e dados padronizados.

    def calculate_reconstruction_error(self, predictions, actual_data):
        return np.mean(np.abs(predictions - actual_data), axis=1)

# Detecção de Anomalias com Isolation Forest:
#    - apply_isolation_forest(): Utiliza Isolation Forest para rotular anomalias 
# com base no erro de reconstrução.

    def apply_isolation_forest(self, reconstruction_error):
	    # Ajuste a contaminação com base nos seus dados
        isolation_forest = IsolationForest(contamination=0.1) 
        return isolation_forest.fit_predict(reconstruction_error.reshape(-1, 1))

# Criação de DataFrames de Resultados e Limpeza:
#    - create_results_dataframe(): Organiza resultados em um DataFrame incluindo 
# tempo, filial, quantidade de item, erro de reconstrução e rótulos de anomalia.
#    - create_cleaned_dataframe(): Cria um DataFrame limpo, substituindo valores 
# de quantidade de item em casos de anomalias.

    def create_results_dataframe(self, reconstruction_error, anomaly_labels):
        self.results_df = pd.DataFrame({
            'time_scale': self.training_df['time_scale'],
            'filial_id': self.training_df['filial_id'],
            'quant_item': self.training_df['quant_item'],
            'Reconstruction_Error': reconstruction_error,
            'Anomaly_Label': anomaly_labels
        })

    def create_cleaned_dataframe(self):
        self.clean_df = self.training_df.copy()
        self.clean_df.loc[self.results_df['Anomaly_Label'] == -1, 'quant_item'] = \
            self.results_df.loc[self.results_df['Anomaly_Label'] == -1, 'quant_item'].copy()

# Salva Resultados e Cria Gráficos Ilustrativos do Processo de Detecção de Ano-
# malias:
#    - save_cleaned_data(): Salva o DataFrame limpo em um arquivo CSV.
#    - plot_inventory_vs_actual(): Gera um gráfico comparativo entre inventário limpo e dados reais.

    def save_cleaned_data(self, output_file='cleaned_inventory.csv'):
        self.clean_df.to_csv(output_file, index=False)

    def plot_inventory_vs_actual(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_df['time_scale'], self.training_df['quant_item'], label='Dados Reais', color='blue')
        plt.plot(self.clean_df['time_scale'], self.clean_df['quant_item'], label='Inventário Limpo', color='red',
                 linestyle='--')
        plt.title('Inventário Limpo vs Dados Reais')
        plt.xlabel('Escala de Tempo')
        plt.ylabel('Quantidade de Itens')
        plt.legend()
        plt.show()

    def plot_overall_trend_mse(self, history):
        keys = list(history.history.keys())
        print(keys)  # Imprime as chaves disponíveis em history.history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[keys[0]])  # Usa a primeira chave disponível
        plt.title('Tendência Geral do Erro Médio Quadrático')
        plt.xlabel('Épocas')
        plt.ylabel('Erro Quadrático Médio')
        plt.show()

    def plot_loss_function(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'])
        plt.title('Função de Perda')
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.show()

# Impressão da Tabela de Resultados:
#    - print_table_of_results(): Imprime uma tabela formatada com os resultados 
# obtidos.

    def print_table_of_results(self):
        print("\nTabela de Resultados:")
        print(tabulate(self.results_df, headers='keys', tablefmt='pretty'))

# Execução Principal:
#    - main(): Orquestra a execução ordenada das funções, proporcionando um flu-
# xo lógico e modular para a análise de inventário.

def main():
    file_path = "/Users/igormol/Downloads/vendas_20180102_20220826/inventory_07813893004.csv"
    inventory_analyzer = InventoryAnalyzer(file_path)

    inventory_analyzer.load_data()
    scaled_data = inventory_analyzer.scale_data()

    # Construir e treinar o modelo CNN
    input_shape = (scaled_data.shape[1], 1)
    cnn_model = inventory_analyzer.build_cnn_model(input_shape)
    history = inventory_analyzer.train_cnn_model(cnn_model, scaled_data.reshape(-1, scaled_data.shape[1], 1), scaled_data)

    # Fazer previsões usando o modelo CNN treinado
    cnn_predictions = cnn_model.predict(scaled_data.reshape(-1, scaled_data.shape[1], 1))

    # Calcular o erro de reconstrução para cada ponto de dados
    reconstruction_error = inventory_analyzer.calculate_reconstruction_error(cnn_predictions, scaled_data)

    # Aplicar Isolation Forest para detecção de anomalias
    anomaly_labels = inventory_analyzer.apply_isolation_forest(reconstruction_error)

    # Criar dataframe de resultados
    inventory_analyzer.create_results_dataframe(reconstruction_error, anomaly_labels)

    # Criar dataframe limpo
    inventory_analyzer.create_cleaned_dataframe()

    # Salvar dados limpos em CSV
    inventory_analyzer.save_cleaned_data()

    # Plotar resultados
    inventory_analyzer.plot_inventory_vs_actual()
    inventory_analyzer.plot_overall_trend_mse(history)
    inventory_analyzer.plot_loss_function(history)

    # Imprimir tabela de resultados
    inventory_analyzer.print_table_of_results()

if __name__ == "__main__":
    main()
