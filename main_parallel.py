import re
import random
import numpy as np
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt
import threading
from time import time
import symengine as sp
import itertools
from multiprocessing import Pool


# Функция для раскрытия одного выражения
def expand_expression(expr):
    expanded = sp.expand(expr)
    return expanded


# Основная функция для параллельного раскрытия
def parallel_expand(expressions):
    with Pool() as pool:
        results = pool.map(expand_expression, expressions)
    return results


class Graph:
    def __init__(self, nodes):
        """Создание простого графа"""
        self.nodes = nodes
        self.list_colors = []
        self.sys_colors = ["#FF0000", "#00FFFF", "#FFFF00", "#0000FF", "#900020", "#808000", "#800080", "#008000"]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(1, nodes + 1))
        self.generate_random_edges()

    def generate_random_edges(self):
        """Метод создания расстановки случайных ребер"""
        edges = []
        for i in range(1, self.nodes + 1):
            for j in range(i + 1, self.nodes + 1):
                if random.random() > 0.6:
                    edges.append((i, j))
        self.graph.add_edges_from(edges)

    def draw_graph(self):
        """Визуализация графа"""
        if len(self.list_colors) != 0:
            nx.draw(self.graph, with_labels=True, node_color=self.list_colors, edge_color='gray', font_weight='bold')
            plt.show()
        else:
            nx.draw(self.graph, with_labels=True, node_color='lightblue', edge_color='gray', font_weight='bold')
            plt.show()

    def get_edges(self):
        """Получение списка ребер"""
        return list(self.graph.edges)

    @staticmethod
    def generate_graph_set(num_graphs, num_nodes):
        """Создание и получение списка длинной num_graphs графов с количеством вершин num_nodes"""
        return {Graph(num_nodes) for _ in range(num_graphs)}

    def incidence_matrix(self):
        """Создание и получение матрицы инцидентности графа"""
        nodes_list = list(self.graph.nodes)
        edges_list = list(self.graph.edges)
        matrix = np.zeros((len(nodes_list), len(edges_list)), dtype=int)

        for edge_index, (u, v) in enumerate(edges_list):
            matrix[nodes_list.index(u)][edge_index] = 1
            matrix[nodes_list.index(v)][edge_index] = 1

        return matrix, {node: i for i, node in enumerate(nodes_list)}

    def method_MAGU(self):
        """Метод Магу-Вейсмана для раскраски графа"""

        def get_dnf(incidence_matrix, node_index):
            dnf_terms = []
            for edge_idx in range(incidence_matrix.shape[1]):
                nodes = [node for node, idx in node_index.items() if incidence_matrix[idx][edge_idx] == 1]
                dnf_terms.append(f'({" + ".join(["x" + str(n) for n in nodes])})')
            return " * ".join(dnf_terms)

        def add_missing(conjunction, all_nodes):
            missing_nodes = all_nodes - conjunction
            return missing_nodes

        def split_expression(expr, max_factors=15):
            if isinstance(expr, str):
                expr = sp.sympify(expr)
            if not isinstance(expr, sp.Mul):
                return [expr]
            factors = list(expr.args)
            split_factors = [factors[i:i + max_factors] for i in range(0, len(factors), max_factors)]
            subexpressions = [sp.Mul(*part) for part in split_factors]
            print(len(subexpressions))
            return subexpressions

        incidence_matrix, node_index = self.incidence_matrix()
        dnf_expression = get_dnf(incidence_matrix, node_index)

        # Разбиваем выражение на подвыражения
        subexpressions = split_expression(dnf_expression, max_factors=6)

        # Замеряем время выполнения
        start_time = time()

        # Раскрываем выражения параллельно
        expanded_expressions = parallel_expand(subexpressions)

        # Умножаем все раскрытые выражения
        multiplied_result = sp.sympify(1)  # Начальное значение для умножения
        for expanded in expanded_expressions:
            multiplied_result = sp.Mul(multiplied_result, expanded)  # Умножаем

        subexpressions = split_expression(multiplied_result, max_factors=6)

        # Умножаем все раскрытые выражения
        multiplied_result = sp.sympify(1)  # Начальное значение для умножения
        for expanded in subexpressions:
            multiplied_result = sp.Mul(multiplied_result, expanded)  # Умножаем

        final_result = sp.expand(multiplied_result)

        # Время выполнения параллельного раскрытия
        print(f"Время выполнения параллельного раскрытия: {time() - start_time:.4f} секунд")

        all_nodes = set(str(i) for i in node_index.keys())
        # print(f"Исходное DNF выражение: {dnf_expression}")
        # start1 = time()
        # expanded_dnf = str(sp.expand(dnf_expression)).split(" + ")
        # print(f"Время последовательного раскрытия: {time() - start1:.4f} секунд")

        expanded_dnf = str(final_result).split(" + ")

        sets = [{str(num) for num in re.findall(r'\d+', s)} for s in expanded_dnf]
        conj = [add_missing(i, all_nodes) for i in sets]
        sort1 = sorted(conj, key=len, reverse=True)

        # Стартуем с хроматического числа
        num_colors = 0
        color_num1 = {}
        while sort1:
            all_nodes_in_conjunction = sort1[0]
            num_colors += 1
            for i in all_nodes_in_conjunction:
                color_num1[i] = self.sys_colors[num_colors % len(self.sys_colors)]
            filtered_data = [s - all_nodes_in_conjunction for s in sort1 if s - all_nodes_in_conjunction]
            sort1 = sorted(filtered_data, key=len, reverse=True)

        self.list_colors = [color_num1.get(str(i), "#FFFFFF") for i in range(1, self.nodes + 1)]

        return num_colors


if __name__ == "__main__":
    """ Метод МАГУ """
    graph = Graph(12)
    chromatic_number = graph.method_MAGU()
    print(f"Хроматическое число графа: {chromatic_number}")
    graph.draw_graph()