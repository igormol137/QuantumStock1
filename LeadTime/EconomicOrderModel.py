# EconomicOrderModel.py
#
# O código a seguir apresenta uma solução para otimização de inventário usando a 
# abordagem de Quantidade Econômica de Pedido (EOQ), Frequência Ótima de Reorde-
# namento e Tempo de Espera Ótimo. A classe InventoryOptimizer encapsula métodos 
# para calcular esses parâmetros com base em dados de estoque, custos de pedido 
# e manutenção. Ao utilizar a classe, é possível carregar dados de inventário de 
# um arquivo CSV, realizar a otimização e exibir uma tabela formatada com as in-
# formações otimizadas.

import pandas as pd
from tabulate import tabulate

# Definimos uma classe chamada InventoryOptimizer para organizar as funcionali-
# dades relacionadas à otimização de inventário.

class InventoryOptimizer:
	
# O método "init" é o construtor da classe. Ele recebe o caminho do arquivo CSV 
# contendo os dados do inventário e carrega esses dados em um DataFrame chamado 
# inventory_data.
	
    def __init__(self, data_path):
        self.inventory_data = pd.read_csv(data_path)

# Este método calcula a Quantidade Econômica de Pedido (EOQ) e a Frequência 
# Ótima de Reordenamento para um item específico com base nos parâmetros de es-
# toque atual, custo de pedido e custo de manutenção.

    def calculate_eoq(self, current_stock, ordering_cost, holding_cost):
        demand = current_stock
        eoq = ((2 * demand * ordering_cost) / holding_cost) ** 0.5
        optimal_reorder_frequency = demand / eoq
        return eoq, optimal_reorder_frequency

# Este método calcula o Tempo de Espera Ótimo com base nos parâmetros de estoque 
# atual, custo de pedido, custo de manutenção e frequência ótima de reordenamento.

    def calculate_optimal_lead_time(self, current_stock, ordering_cost, holding_cost, reorder_frequency):
        demand = current_stock
        eoq, optimal_reorder_frequency = self.calculate_eoq(current_stock, ordering_cost, holding_cost)
        optimal_lead_time = reorder_frequency / optimal_reorder_frequency
        return optimal_lead_time

# Este método otimiza o inventário ao calcular a EOQ, a Frequência Ótima de 
# Reordenamento e o Tempo de Espera Ótimo para cada linha no DataFrame 
# inventory_data.

    def optimize_inventory(self):
        # Calcula EOQ:
        self.inventory_data['EOQ'], self.inventory_data['Optimal Reorder Frequency'] = zip(*self.inventory_data.apply(
            lambda row: self.calculate_eoq(row['Current Stock'], row['Ordering Cost'], row['Holding Cost']), axis=1
        ))

        # Calcula Tempo de Cobertura Ótimo:
        self.inventory_data['Optimal Lead Time'] = self.inventory_data.apply(
            lambda row: self.calculate_optimal_lead_time(row['Current Stock'], row['Ordering Cost'],
                                                         row['Holding Cost'], row['Reorder Frequency']), axis=1
        )

# Este método exibe uma tabela formatada com as colunas selecionadas, incluindo 
# ID do produto, Frequência de Reordenamento, EOQ, Frequência Ótima de Reordena-
# mento e Tempo de Espera Ótimo.

    def display_optimized_table(self):
        # Print the improved table
        selected_columns = ['Product ID', 'Reorder Frequency', 'EOQ', 'Optimal Reorder Frequency', 'Optimal Lead Time']
        print(tabulate(self.inventory_data[selected_columns], headers='keys', tablefmt='pretty'))


# Aqui, estamos instanciando a classe InventoryOptimizer, carregando os dados do 
# inventário, otimizando o inventário e exibindo a tabela otimizada. Substitua 
# "your_inventory_data.csv" pelo caminho real ou URL do seu arquivo CSV.

if __name__ == "__main__":
    data_path = "/Users/igormol/Desktop/QuantumStock/LeadTime/random_inventory_data.csv"
    optimizer = InventoryOptimizer(data_path)
    optimizer.optimize_inventory()
    optimizer.display_optimized_table()
