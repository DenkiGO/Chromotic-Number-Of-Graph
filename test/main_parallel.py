import re
import multiprocessing
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
from multiprocessing.pool import ThreadPool


# Функция для раскрытия одного выражения
def expand_expression(expr):
    expanded = sp.expand(expr)
    return expanded


# Основная функция для параллельного раскрытия
def parallel_expand(expressions):
    with ThreadPool(processes=4) as pool:
        results = pool.map(expand_expression, expressions)
    return results


class Graph:
    def __init__(self, nodes, edges=None, max_edges=None):
        """
        Создание простого графа.
        :param nodes: Количество вершин.
        :param edges: Количество рёбер. Если None, рёбра генерируются случайно.
        :param max_edges: Максимальное допустимое количество рёбер. Если None, ограничение отсутствует.
        """
        self.nodes = nodes
        self.edges = edges
        self.max_edges = max_edges
        self.list_colors = []
        self.sys_colors = ["#FF0000", "#00FFFF", "#FFFF00", "#0000FF", "#900020", "#808000", "#800080", "#008000"]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(1, nodes + 1))
        self.generate_random_edges()

    def generate_random_edges(self):
        """Метод создания случайных рёбер с указанным количеством"""
        max_possible_edges = self.nodes * (self.nodes - 1) // 2  # Максимальное количество рёбер в графе

        if self.edges is not None:
            # Если количество рёбер указано, проверяем, что оно не превышает max_edges (если задан)
            if self.max_edges is not None and self.edges > self.max_edges:
                raise ValueError(f"Невозможно создать {self.edges} рёбер. "
                                f"Максимальное допустимое количество рёбер: {self.max_edges}")
            if self.edges > max_possible_edges:
                raise ValueError(f"Невозможно создать {self.edges} рёбер для графа с {self.nodes} вершинами. "
                               f"Максимальное количество рёбер: {max_possible_edges}")

            # Генерация случайных рёбер
            all_possible_edges = [(i, j) for i in range(1, self.nodes + 1) for j in range(i + 1, self.nodes + 1)]
            selected_edges = random.sample(all_possible_edges, self.edges)
            self.graph.add_edges_from(selected_edges)
        else:
            # Если количество рёбер не указано, генерируем рёбра случайно
            if self.max_edges is not None:
                # Если задан max_edges, выбираем случайное количество рёбер от 0 до max_edges
                random_edge_count = random.randint(0, min(self.max_edges, max_possible_edges))
                all_possible_edges = [(i, j) for i in range(1, self.nodes + 1) for j in range(i + 1, self.nodes + 1)]
                selected_edges = random.sample(all_possible_edges, random_edge_count)
                self.graph.add_edges_from(selected_edges)
            else:
                # Если max_edges не задан, генерируем рёбра случайно без ограничений
                for i in range(1, self.nodes + 1):
                    for j in range(i + 1, self.nodes + 1):
                        if random.random() > 0.6:
                            self.graph.add_edge(i, j)

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
    def generate_graph_set(num_graphs, num_nodes, edges=None, max_edge=None):
        """Создание и получение списка длинной num_graphs графов с количеством вершин num_nodes"""
        return {Graph(num_nodes, edges, max_edge) for _ in range(num_graphs)}

    def incidence_matrix(self):
        """Создание и получение матрицы инцидентности графа"""
        nodes_list = list(self.graph.nodes)
        edges_list = list(self.graph.edges)
        matrix = np.zeros((len(nodes_list), len(edges_list)), dtype=int)

        for edge_index, (u, v) in enumerate(edges_list):
            matrix[nodes_list.index(u)][edge_index] = 1
            matrix[nodes_list.index(v)][edge_index] = 1

        return matrix, {node: i for i, node in enumerate(nodes_list)}

    def method_MAGU_usual(self, dnf_ex=None, node_ind=None):
        """Метод Магу послед."""
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

        if dnf_ex:
            dnf_expression = dnf_ex
            node_index = node_ind
        else:
            incidence_matrix, node_index = self.incidence_matrix()
            dnf_expression = get_dnf(incidence_matrix, node_index)

        count_dnf = len(dnf_expression.split("*"))

        all_nodes = set(str(i) for i in node_index.keys())
        # print(f"Исходное DNF выражение: {dnf_expression}")
        start1 = time()
        expanded_dnf = str(sp.expand(dnf_expression)).split(" + ")

        res_time = f"{time() - start1:.4f}"
        print(f"Время последовательного раскрытия: {res_time} секунд")

        with open('paralell_and_usual_11_all.txt', 'a') as file:
            dict_usual = {}
            dict_usual["method"] = "usual"
            dict_usual["time"] = res_time
            dict_usual["count_dnf"] = count_dnf
            dict_usual["dnf_expression"] = dnf_expression
            file.write(str(dict_usual)+"\n")

    def method_MAGU(self, dnf_ex=None, node_ind=None):
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
            return subexpressions

        if dnf_ex:
            dnf_expression = dnf_ex
            node_index = node_ind
        else:
            incidence_matrix, node_index = self.incidence_matrix()
            dnf_expression = get_dnf(incidence_matrix, node_index)

        count_dnf = len(dnf_expression.split("*"))

        # Разбиваем выражение на подвыражения
        subexpressions = split_expression(dnf_expression, max_factors=4)

        # Замеряем время выполнения
        start_time = time()

        # sp.simplify(expr)

        # Раскрываем выражения параллельно
        expanded_expressions = parallel_expand(subexpressions)

        # Умножаем все раскрытые выражения
        multiplied_result = sp.sympify(1)  # Начальное значение для умножения
        for expanded in expanded_expressions:
            multiplied_result = sp.Mul(multiplied_result, expanded)  # Умножаем

        subexpressions = split_expression(multiplied_result, max_factors=2)
        expanded_expressions = parallel_expand(subexpressions)

        # Умножаем все раскрытые выражения
        multiplied_result = sp.sympify(1)  # Начальное значение для умножения
        for expanded in expanded_expressions:
            multiplied_result = sp.Mul(multiplied_result, expanded)  # Умножаем

        final_result = sp.expand(multiplied_result)

        # Время выполнения параллельного раскрытия
        res_time = f"{time() - start_time:.4f}"
        print(f"Время выполнения параллельного раскрытия: {res_time} секунд")

        with open('paralell_and_usual_11_all.txt', 'a') as file:
            dict_parallel = {}
            dict_parallel["method"] = "parallel"
            dict_parallel["time"] = res_time
            dict_parallel["count_dnf"] = count_dnf
            dict_parallel["dnf_expression"] = dnf_expression
            file.write(str(dict_parallel)+'\n')



def run_method_MAGU_usual(graph):
    graph.method_MAGU_usual()


def run_method_MAGU(graph):
    graph.method_MAGU()


if __name__ == "__main__":
    for i in range(50):
        print(f"---------------------|| {i} ||---------------------")
        # Генерируем набор графов
        graphs = Graph.generate_graph_set(1, 11)

        with multiprocessing.Pool(processes=2) as pool:
            # Запускаем методы для каждого графа
            pool.map(run_method_MAGU, graphs)
            pool.map(run_method_MAGU_usual, graphs)
        print()

    print("Все методы завершены.")