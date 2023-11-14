# Random_Forests.py
#
# O código-fonte a seguir é uma implementação de um otimizador de inventário. 
# O sistema proposto emprega uma abordagem de Floresta Aleatória para prever a 
# quantidade ótima de itens em estoque com base em dados históricos de vendas.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

class OtimizadorInventario:
    def __init__(self, dataframe):
        self.dataframe = dataframe

# O método dividir_treino_teste extrai as features e o target do dataframe, uti-
# lizando as colunas 'time_scale' e 'filial_id' como features, e 'quant_item' 
# como o target. A função train_test_split da biblioteca scikit-learn é então 
# aplicada para separar os dados em conjuntos de treino e teste, respeitando uma 
# proporção predefinida.

    def dividir_treino_teste(self):
        X = self.dataframe[['time_scale', 'filial_id']]
        y = self.dataframe['quant_item']
        return train_test_split(X, y, test_size=0.2, random_state=42)

# O método treinar_random_forest implementa o treinamento do modelo de Floresta 
# Aleatória. Utilizando a classe RandomForestRegressor do scikit-learn, o modelo 
# é instanciado e treinado com os dados de treino fornecidos. O número de árvo-
# res na floresta é um hiperparâmetro ajustável, sendo 100 o valor padrão.

    def treinar_random_forest(self, X_treino, y_treino, n_estimators=100):
        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf_model.fit(X_treino, y_treino)
        return rf_model

# O método avaliar_modelo avalia o desempenho do modelo treinado. Realiza pre-
# visões sobre os dados de teste e calcula o desvio padrão e o erro estatístico 
# (MSE) comparando as previsões com os valores reais.

    def avaliar_modelo(self, modelo, X_teste, y_teste):
        y_pred = modelo.predict(X_teste)
        desvio_padrao = np.std(y_teste)
        erro_estatistico = mean_squared_error(y_teste, y_pred, squared=False)
        return y_pred, desvio_padrao, erro_estatistico

# As funções plotar_resultados e plotar_tendencia_mse são empregadas para criar 
# visualizações informativas. A primeira gera um gráfico de dispersão comparando 
# os valores reais com os valores previstos, enquanto a segunda apresenta a ten-
# dência geral do MSE em relação ao número de árvores na floresta aleatória.

def plotar_resultados(X_teste, y_teste, y_pred, titulo):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_teste['time_scale'], y_teste, label='Valores reais', alpha=0.7, marker='o')
    plt.scatter(X_teste['time_scale'], y_pred, label='Valores previstos', alpha=0.7, marker='x')
    plt.xlabel('Escala de Tempo')
    plt.ylabel('Quantidade de Itens')
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.show()

def plotar_tendencia_mse(valores_mse):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101), valores_mse, marker='o')
    plt.xlabel('Número de Árvores na Floresta Aleatória')
    plt.ylabel('Erro Quadrático Médio (MSE)')
    plt.title('Tendência Geral do MSE com Diferentes Números de Árvores')
    plt.grid(True)
    plt.show()

# O bloco a seguir orquestra a execução do programa. O CSV contendo os dados de 
# inventário é carregado no dataframe. Uma instância OtimizadorInventario é cri-
# ada, e o processo de treinamento e avaliação do modelo é iniciado. Os resulta-
# dos são tabulados e visualizados, proporcionando uma compreensão abrangente do 
# desempenho do modelo.

if __name__ == "__main__":
    # Load CSV file into the dataframe
    arquivo_csv = "/Users/igormol/Downloads/vendas_20180102_20220826/inventory_07813893004.csv"
    dataframe_inventario = pd.read_csv(arquivo_csv)

    otimizador = OtimizadorInventario(dataframe_inventario)
    X_treino, X_teste, y_treino, y_teste = otimizador.dividir_treino_teste()

    # Treinar e avaliar o modelo de Floresta Aleatória
    modelo_rf = otimizador.treinar_random_forest(X_treino, y_treino)
    y_pred, desvio_padrao, erro_estatistico = otimizador.avaliar_modelo(modelo_rf, X_teste, y_teste)

    # Criar um DataFrame para armazenar valores reais, previstos, desvio padrão e erro estatístico
    resultados_df = pd.DataFrame({
        'Escala de Tempo': X_teste['time_scale'],
        'Filial ID': X_teste['filial_id'],
        'Quantidade Real': y_teste,
        'Quantidade Prevista': y_pred,
        'Desvio Padrão': desvio_padrao,
        'Erro Estatístico': erro_estatistico
    })

    # Imprimir os resultados em uma tabela formatada
    print(tabulate(resultados_df, headers='keys', tablefmt='pretty', showindex=False))

    # Plotar os resultados para um modelo específico de Floresta Aleatória
    plotar_resultados(X_teste, y_teste, y_pred, 'Quantidade Real vs. Quantidade Prevista de Itens')

    # Calcular MSE para diferentes previsões
    valores_mse = []
    for i in range(1, 101):  # Número de árvores na floresta
        modelo_rf = otimizador.treinar_random_forest(X_treino, y_treino, n_estimators=i)
        y_pred = modelo_rf.predict(X_teste)
        mse = mean_squared_error(y_teste, y_pred)
        valores_mse.append(mse)

    # Plotar a tendência geral do MSE
    plotar_tendencia_mse(valores_mse)
