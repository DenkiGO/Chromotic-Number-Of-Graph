import re
import random
import numpy as np
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt
import threading
from time import time
# import sympy as sp
import symengine as sp
import itertools


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
                if random.random() > 0.6:  # 50% вероятность создания ребра
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
        # Получаем матрицу инцидентности графа

        def get_dnf(incidence_matrix, node_index):
            dnf_terms = []
            for edge_idx in range(incidence_matrix.shape[1]):
                nodes = [node for node, idx in node_index.items() if incidence_matrix[idx][edge_idx] == 1]
                dnf_terms.append(f'({" + ".join(["x" + str(n) for n in nodes])})')
            return " * ".join(dnf_terms)

        def add_missing(conjunction, all_nodes):
            missing_nodes = all_nodes - conjunction
            return missing_nodes

        incidence_matrix, node_index = self.incidence_matrix()

        # Получаем ДНФ для этого графа
        dnf_expression = get_dnf(incidence_matrix, node_index)

        all_nodes = set(str(i) for i in node_index.keys())

        # Функция для выполнения второго задания
        def task2(dnf_expression):
            global expandewew_dnf
            print(dnf_expression)
            start1 = time()
            expandewew_dnf = str(sp.expand(dnf_expression)).split(" + ")
            end1 = time()
            print(f"Время - 2: {end1 - start1}")

        # Создаем потоки
        thread2 = threading.Thread(target=task2, args=(dnf_expression,))

        thread2.start()

        thread2.join()

        sets = [{str(num) for num in re.findall(r'\d+', s)} for s in expandewew_dnf]
        conj = [add_missing(i, all_nodes) for i in sets]
        sort1 = sorted(conj, key=len, reverse=True)
        # Стартуем с хроматического числа
        num_colors = 0

        color_num1 = {}
        while sort1:
            all_nodes_in_conjunction = sort1[0]
            num_colors += 1

            for i in all_nodes_in_conjunction:
                color_num1[i] = self.sys_colors[num_colors]

            # Фильтруем оставшиеся конъюнкции, удаляя вершины из текущего множества
            filtered_data = [s - all_nodes_in_conjunction for s in sort1 if s - all_nodes_in_conjunction]

            # Сортируем оставшиеся множества по убыванию длины
            sort1 = sorted(filtered_data, key=len, reverse=True)

        self.list_colors = [color_num1[str(i)] for i in range(1, self.nodes + 1)]

        return num_colors

    def greedy_coloring(self):
        """
        Жадная раскраска графа (сортировка по убыванию степени вершины).
        Возвращает приближенное хроматическое число.
        """
        # Сортируем вершины по убыванию степени
        sorted_nodes = sorted(self.graph.nodes, key=lambda x: self.graph.degree[x], reverse=True)
        color_assignment = {}

        for node in sorted_nodes:
            # Собираем цвета всех соседей
            used_colors = set()
            for neighbor in self.graph.neighbors(node):
                if neighbor in color_assignment:
                    used_colors.add(color_assignment[neighbor])

            # Находим минимальный доступный цвет
            color = 0
            while color in used_colors:
                color += 1

            color_assignment[node] = color

        # Преобразуем в цвета из sys_colors (с повторением, если цветов не хватает)
        self.list_colors = [
            self.sys_colors[color_assignment[node] % len(self.sys_colors)]
            for node in range(1, self.nodes + 1)
        ]

        # Хроматическое число = максимальный цвет + 1 (т.к. цвета нумеруются с 0)
        chromatic_number = max(color_assignment.values()) + 1 if color_assignment else 0
        return chromatic_number

    def genetic_algorithm_coloring(self, population_size=50, generations=100, mutation_rate=0.1):
        """
        Генетический алгоритм для поиска хроматического числа.
        :param population_size: Размер популяции.
        :param generations: Количество поколений.
        :param mutation_rate: Вероятность мутации.
        :return: Приближенное хроматическое число.
        """
        def fitness(individual):
            """Оценка приспособленности раскраски (минимизация конфликтов)."""
            conflicts = 0
            for u, v in self.graph.edges:
                if individual[u - 1] == individual[v - 1]:
                    conflicts += 1
            return -conflicts  # Чем меньше конфликтов, тем лучше

        def crossover(parent1, parent2):
            """Одноточечный кроссовер."""
            point = random.randint(1, self.nodes - 1)
            child = parent1[:point] + parent2[point:]
            return child

        def mutate(individual):
            """Мутация: случайное изменение цвета одной вершины."""
            if random.random() < mutation_rate:
                idx = random.randint(0, self.nodes - 1)
                individual[idx] = random.randint(0, self.nodes - 1)
            return individual

        # Инициализация популяции
        population = []
        for _ in range(population_size):
            individual = [random.randint(0, self.nodes - 1) for _ in range(self.nodes)]
            population.append(individual)

        # Основной цикл генетического алгоритма
        for generation in range(generations):
            # Оценка приспособленности
            fitness_scores = [fitness(individual) for individual in population]

            # Селекция (турнирный отбор)
            new_population = []
            for _ in range(population_size):
                # Выбираем двух случайных особей
                candidates = random.sample(range(population_size), 2)
                # Выбираем лучшую
                winner = candidates[0] if fitness_scores[candidates[0]] > fitness_scores[candidates[1]] else candidates[1]
                new_population.append(population[winner])

            # Кроссовер
            for i in range(0, population_size, 2):
                if i + 1 < population_size:
                    parent1, parent2 = new_population[i], new_population[i + 1]
                    child1 = crossover(parent1, parent2)
                    child2 = crossover(parent2, parent1)
                    new_population[i], new_population[i + 1] = child1, child2

            # Мутация
            for i in range(population_size):
                new_population[i] = mutate(new_population[i])

            # Обновление популяции
            population = new_population

        # Выбор лучшей раскраски
        best_individual = max(population, key=fitness)
        self.list_colors = [self.sys_colors[color % len(self.sys_colors)] for color in best_individual]

        # Хроматическое число = количество уникальных цветов в лучшей раскраске
        chromatic_number = len(set(best_individual))
        return chromatic_number


""" Метод МАГУ """
graph = Graph(10)
chromatic_number = graph.method_MAGU()
print(f"Хроматическое число графа: {chromatic_number}")
graph.draw_graph()

""" Жадный алгоритм """
g = Graph(20)
print("Приближенное хроматическое число:", g.greedy_coloring())
g.draw_graph()

""" Генетический алгоритм """
g = Graph(20)
print("Приближенное хроматическое число (генетический алгоритм):", g.genetic_algorithm_coloring())
g.draw_graph()


# all_graph = Graph.generate_graph_set(5, 11)
#
# for gr in all_graph:
#     gr.method_MAGU()

