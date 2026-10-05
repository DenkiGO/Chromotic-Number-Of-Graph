import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import symengine as sp
import re
from time import time
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
    def __init__(self, nodes, edges=None, edges_procent=0.6):
        """
        Создание простого графа.
        :param nodes: Количество вершин.
        :param edges: Количество рёбер. Если None, рёбра генерируются случайно.
        """
        self.nodes = nodes
        self.edges = edges
        if edges:
            self.edge_count = edges
        else:
            self.edge_count = 0
        self.list_colors = []
        self.sys_colors = ["#FF0000", "#00FFFF", "#FFFF00", "#0000FF", "#900020", "#808000", "#800080", "#008000"]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(1, nodes + 1))
        self.edges_procent = edges_procent
        self.generate_random_edges()

    def generate_random_edges(self):
        """Метод создания случайных рёбер с указанным количеством"""
        if self.edges is None:
            self.edge_count = 0  # Инициализация счётчика рёбер
            # Если количество рёбер не указано, генерируем рёбра случайно
            for i in range(1, self.nodes + 1):
                for j in range(i + 1, self.nodes + 1):
                    if random.random() < self.edges_procent:
                        self.graph.add_edge(i, j)
                        self.edge_count += 1  # Увеличиваем счётчик при добавлении ребра
        else:
            # Если количество рёбер указано, генерируем ровно столько рёбер
            max_possible_edges = self.nodes * (self.nodes - 1) // 2  # Максимальное количество рёбер в графе
            if self.edges > max_possible_edges:
                raise ValueError(f"Невозможно создать {self.edges} рёбер для графа с {self.nodes} вершинами. "
                               f"Максимальное количество рёбер: {max_possible_edges}")

            # Генерация случайных рёбер
            all_possible_edges = [(i, j) for i in range(1, self.nodes + 1) for j in range(i + 1, self.nodes + 1)]
            selected_edges = random.sample(all_possible_edges, self.edges)
            self.graph.add_edges_from(selected_edges)

    def draw_graph(self, ax=None):
        pos = nx.circular_layout(self.graph)
        if ax is None:
            ax = plt.gca()  # Используется стандартная ось, если ax не передан
        if len(self.list_colors) != 0:
            nx.draw(self.graph, pos, ax=ax, with_labels=True,
                    node_color=self.list_colors, edge_color='gray', font_weight='bold')
        else:
            nx.draw(self.graph, pos, ax=ax, with_labels=True,
                    node_color='lightblue', edge_color='gray', font_weight='bold')
        if ax is None:
            plt.show()

    def get_edges(self):
        """Получение списка ребер"""
        return list(self.graph.edges)

    @staticmethod
    def generate_graph_set(num_graphs, num_nodes, edges=None):
        """Создание и получение списка длинной num_graphs графов с количеством вершин num_nodes"""
        return {Graph(num_nodes, edges) for _ in range(num_graphs)}

    def incidence_matrix(self):
        """Создание и получение матрицы инцидентности графа"""
        nodes_list = list(self.graph.nodes)
        edges_list = list(self.graph.edges)
        matrix = np.zeros((len(nodes_list), len(edges_list)), dtype=int)

        for edge_index, (u, v) in enumerate(edges_list):
            matrix[nodes_list.index(u)][edge_index] = 1
            matrix[nodes_list.index(v)][edge_index] = 1

        return matrix, {node: i for i, node in enumerate(nodes_list)}

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

        def comp(dnf_expression):
            # Разбиваем выражение на подвыражения
            subexpressions = split_expression(dnf_expression, max_factors=4)

            # Замеряем время выполнения
            start_time = time()

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

            expanded_dnf = str(final_result).split(" + ")

            cleaned_lst = [re.sub(r'\*\*\d+', '', s) for s in expanded_dnf]
            cleaned_lst = [re.sub(r'^\d+\*', '', s) for s in cleaned_lst]

            result = [s.split('*') for s in cleaned_lst]

            min_len = min(len(lst) for lst in result)
            filtered_lists = [lst for lst in result if len(lst) == min_len]

            return filtered_lists, min_len

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

        all_nodes = set(str(i) for i in node_index.keys())
        # print(f"Исходное DNF выражение: {dnf_expression}")
        # start1 = time()
        # expanded_dnf = str(sp.expand(dnf_expression)).split(" + ")
        # print(f"Время последовательного раскрытия: {time() - start1:.4f} секунд")

        expanded_dnf = str(final_result).split(" + ")

        cleaned_lst = [re.sub(r'\*\*\d+', '', s) for s in expanded_dnf]
        cleaned_lst = [re.sub(r'^\d+\*', '', s) for s in cleaned_lst]

        sets1 = [{str(num) for num in re.findall(r'\d+', s)} for s in cleaned_lst]
        conj1 = [add_missing(i, all_nodes) for i in sets1]

        hashable_conj1 = [frozenset(s) for s in conj1]
        unique_conj1 = list(dict.fromkeys(hashable_conj1))
        unique_conj1 = [set(s) for s in unique_conj1]
        my_list1 = sorted(unique_conj1, key=len, reverse=True)

        sort1 = [item for item in my_list1 if item != set()]

        # Время выполнения параллельного раскрытия
        res_time_1 = f"{time() - start_time:.4f}"
        print(f"Время выполнения параллельного раскрытия 2: {res_time_1} секунд")

        dict_1 = {}

        for node in self.graph.nodes:
            list_1 = []
            for j in range(len(sort1)):
                if str(node) in sort1[j]:
                    list_1.append(j)
            dict_1[str(node)] = list_1

        parts = []
        for k, v in dict_1.items():
            y_terms = [f"y{i}" for i in v]
            term = "+".join(y_terms)
            if len(y_terms) > 1:
                term = f"({term})"
            parts.append(term)
        result = "*".join(parts)

        res = comp(result)

        count_color = len(res[0])

        sum_color = res[1]

        # # Стартуем с хроматического числа
        # num_colors = 0
        # color_num1 = {}
        # while sort1:
        #     all_nodes_in_conjunction = sort1[0]
        #     num_colors += 1
        #     for i in all_nodes_in_conjunction:
        #         color_num1[i] = self.sys_colors[num_colors % len(self.sys_colors)]
        #     filtered_data = [s - all_nodes_in_conjunction for s in sort1 if s - all_nodes_in_conjunction]
        #     sort1 = sorted(filtered_data, key=len, reverse=True)
        #
        # self.list_colors = [color_num1.get(str(i), "#FFFFFF") for i in range(1, self.nodes + 1)]

        # Время выполнения параллельного раскрытия
        res_time_2 = f"{time() - start_time:.4f}"
        print(f"Время выполнения параллельного раскрытия 3: {res_time_2} секунд")

        with open('test_all_methods.txt', 'a') as file:
            dict_usual = {}
            dict_usual["method"] = "MAGU"
            dict_usual["time"] = res_time_2
            dict_usual["nodes"] = self.nodes
            dict_usual["edges"] = self.edge_count
            dict_usual["result"] = sum_color
            file.write(str(dict_usual) + "\n")

        return sum_color


    def greedy_coloring(self):
        """
        Жадная раскраска графа (сортировка по убыванию степени вершины).
        Возвращает приближенное хроматическое число.
        """
        start_time = time()
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
        res_time = f"{time() - start_time:.4f}"
        # print(f"Жадный алгоритм завершен за {res_time}")

        with open('test_all_methods.txt', 'a') as file:
            dict_usual = {}
            dict_usual["method"] = "Жадный"
            dict_usual["time"] = res_time
            dict_usual["nodes"] = self.nodes
            dict_usual["edges"] = self.edge_count
            dict_usual["result"] = chromatic_number
            file.write(str(dict_usual)+"\n")

        return dict_usual

    def genetic_algorithm_coloring(self, population_size=50, generations=100, mutation_rate=0.9, crossover_rate=0.9,
                                   coloring_method='Случайный', povt_count=50):
        """
        Генетический алгоритм с отслеживанием минимального хроматического числа во всех поколениях.
        :param population_size: Размер популяции.
        :param generations: Количество поколений.
        :param mutation_rate: Вероятность мутации.
        :param crossover_rate: Вероятность кроссовера.
        :param coloring_method: Метод раскраски ('Жадный', 'Нез. множ.', 'Случайный', 'DSATUR').
        :return: Минимальное хроматическое число, лучшая раскраска и данные для визуализации.
        """

        # Одноточечный кроссовер
        def crossover(parent1, parent2, crossover_point):
            """
            Одноточечный кроссовер: создает потомка, комбинируя цвета вершин из двух родителей.
            :param parent1: Список цветов первой родительской особи.
            :param parent2: Список цветов второй родительской особи.
            :return: Список цветов потомка.
            """
            n = len(parent1)
            child = parent1.copy()
            for i in range(crossover_point, n):
                child[i] = parent2[i]
            return child

        # Новый кроссовер
        def perform_crossover(new_population, crossover_probability, graph):
            """
            Выполняет кроссовер для каждой особи с вероятностью crossover_probability.
            Если кроссовер выполняется, выбирается случайный второй родитель, создаются два потомка,
            и первый родитель заменяется лучшим потомком.
            :param new_population: Список особей (раскрасок).
            :param crossover_probability: Вероятность кроссовера для каждой особи (например, 0.1).
            :param graph: Граф (объект networkx.Graph) для вычисления приспособленности.
            :return: Обновленная популяция.
            """
            population_size = len(new_population)

            # print("Кроссовер:")

            for i in range(population_size):
                if random.random() < crossover_probability:
                    # Выбираем случайного второго родителя (не равного текущему)
                    other_indices = [j for j in range(population_size) if j != i]
                    if not other_indices:  # Если нет других особей
                        continue
                    parent2_index = random.choice(other_indices)
                    parent1 = new_population[i]
                    parent2 = new_population[parent2_index]

                    n = len(parent1)
                    crossover_point = random.randint(1, n - 1)

                    # Создаем двух потомков
                    child1 = crossover(parent1, parent2, crossover_point)
                    child2 = crossover(parent2, parent1, crossover_point)

                    fitness_child1 = fitness(child1)
                    fitness_child2 = fitness(child2)
                    fitness_parent1 = fitness(parent1)

                    best_child = child1 if fitness_child1 > fitness_child2 else child2
                    best_fitness = max(fitness_child1, fitness_child2)
                    # print('---')
                    # print(f"особь №{i}. Родитель 1: {parent1} ({fitness_parent1}), Родитель 2: {parent2} | точка кроссовера: {crossover_point}\nРебенок 1: {child1} ({fitness_child1}), Ребенок 2: {child2} ({fitness_child2}) | Лучший: {best_child} ({max(fitness_child1, fitness_child2, fitness_parent1)})")

                    # Заменяем только если потомок лучше родителя
                    if best_fitness > fitness_parent1:
                        new_population[i] = best_child

                    # Заменяем первого родителя (текущую особь) лучшим потомком
                    new_population[i] = best_child
            # print()
            return new_population

        def fitness(individual):
            """Оценка приспособленности: минимизация конфликтов и числа цветов."""
            conflicts = 0
            for u, v in self.graph.edges:
                if individual[u - 1] == individual[v - 1]:
                    conflicts += 1
            num_colors = len(set(individual))
            return -conflicts - 1.0 * num_colors  # Увеличенный штраф за цвета

        def mutate(individual, graph, mutation_rate, count_osob):
            """
            Мутация с выбором цвета из существующих или неконфликтного.
            :param individual: Список цветов особи.
            :param graph: Граф (объект networkx.Graph).
            :param mutation_rate: Вероятность мутации.
            :return: Лучшая особь (исходная или мутированная).
            """
            ind = individual.copy()

            if random.random() < mutation_rate:
                # Выбираем случайную вершину
                idx = random.randint(0, len(individual) - 1)
                # Создаем список смежности
                adj_list = {i: set() for i in range(1, len(individual) + 1)}
                for u, v in graph.edges:
                    adj_list[u].add(v)
                    adj_list[v].add(u)
                # Определяем цвета соседей
                used_colors = {individual[u - 1] for u in adj_list[idx + 1]}
                # Определяем текущие цвета
                current_colors = list(set(individual))
                # Формируем доступные цвета
                available_colors = [c for c in current_colors if c not in used_colors]
                if not available_colors or (len(available_colors) == 1 and available_colors[0] == individual[idx]):  # Если нет существующих неконфликтных цветов
                    available_colors = [len(current_colors)]
                color = random.choice(available_colors)
                individual[idx] = color
                # print("---")
                # print(f"особь №{count_osob}. Мутация в вершине: {idx}. Новый цвет: {color}\n До {ind} | После {individual}")

            return individual

        def greedy_coloring(graph, nodes):
            """
            Жадная раскраска графа с проверкой на конфликты.
            :param graph: Граф (объект networkx.Graph).
            :param nodes: Количество вершин.
            :return: Список цветов для вершин (без конфликтов).
            :raises ValueError: Если обнаружены конфликты в раскраске.
            """
            # 1. Создание списка смежности
            adj_list = {i: set() for i in range(1, nodes + 1)}
            for u, v in graph.edges:
                adj_list[u].add(v)
                adj_list[v].add(u)
            # 2. Инициализация цветов (-1 означает нераскрашенную вершину)
            colors = [-1] * nodes
            # 3. Случайный порядок вершин
            random_order = list(range(1, nodes + 1))
            random.shuffle(random_order)
            # 4. Жадная раскраска
            for vertex in random_order:
                # Собираем цвета уже раскрашенных соседей
                used_colors = {colors[v - 1] for v in adj_list[vertex] if colors[v - 1] != -1}
                # Находим минимальный неконфликтный цвет
                color = 0
                while color in used_colors:
                    color += 1
                colors[vertex - 1] = color
            # 5. Проверка на конфликты
            for u, v in graph.edges:
                if colors[u - 1] == colors[v - 1]:
                    raise ValueError(f"Конфликт обнаружен на ребре ({u}, {v}): оба имеют цвет {colors[u - 1]}")
            return colors

        def independent_set_coloring(graph, nodes):
            """
            Раскраска графа с использованием жадного выбора независимых множеств.
            Вершины группируются в независимые множества (без смежных вершин), каждому множеству назначается уникальный цвет.
            :param graph: Граф (объект networkx.Graph), содержащий рёбра.
            :param nodes: Количество вершин в графе (нумерация с 1).
            :return: Список цветов (индексы от 0), где colors[i] — цвет вершины i+1.
            """
            # Инициализация списка цветов: -1 означает, что вершина не раскрашена
            colors = [-1] * nodes
            # Создание списка смежности: для каждой вершины хранится множество её соседей
            adj_list = {i: set() for i in range(1, nodes + 1)}
            for u, v in graph.edges:
                adj_list[u].add(v)
                adj_list[v].add(u)
            # Начальный цвет
            color = 0
            # Множество оставшихся (нераскрашенных) вершин
            remaining_vertices = set(range(1, nodes + 1))
            # Продолжаем, пока есть нераскрашенные вершины
            while remaining_vertices:
                # Инициализация независимого множества
                independent_set = []
                # Случайный порядок вершин для выбора
                available = list(remaining_vertices)
                random.shuffle(available)
                # Жадный выбор вершин для независимого множества
                for v in available:
                    # Проверяем, что вершина v не смежна с уже выбранными вершинами
                    if all(u not in independent_set for u in adj_list[v]):
                        independent_set.append(v)
                # Назначаем текущий цвет всем вершинам в независимом множестве
                for v in independent_set:
                    colors[v - 1] = color
                    # Удаляем раскрашенные вершины из оставшихся
                    remaining_vertices.remove(v)
                # Переходим к следующему цвету
                color += 1
            return colors

        def random_safe_coloring(graph, nodes):
            """
            Случайная раскраска графа, гарантирующая отсутствие конфликтов.
            Каждой вершине назначается минимальный возможный цвет, не используемый её соседями.
            :param graph: Граф (объект networkx.Graph), содержащий рёбра.
            :param nodes: Количество вершин в графе (нумерация с 1).
            :return: Список цветов (индексы от 0), где colors[i] — цвет вершины i+1.
            """
            # Создание списка смежности: для каждой вершины хранится множество её соседей
            adj_list = {i: set() for i in range(1, nodes + 1)}
            for u, v in graph.edges:
                adj_list[u].add(v)
                adj_list[v].add(u)
            # Оценка максимальной степени графа для начального числа цветов
            max_degree = max(len(adj_list[v]) for v in adj_list)
            k = max_degree + 1  # Верхняя граница числа цветов
            # Инициализация списка цветов: -1 означает, что вершина не раскрашена
            colors = [-1] * nodes
            # Случайный порядок вершин
            vertices = list(range(1, nodes + 1))
            random.shuffle(vertices)
            # Раскраска каждой вершины
            for v in vertices:
                # Собираем цвета уже раскрашенных соседей
                used_colors = {colors[u - 1] for u in adj_list[v] if colors[u - 1] != -1}
                # Формируем список доступных цветов (от 0 до k-1, не в used_colors)
                available_colors = [c for c in range(k) if c not in used_colors]
                # Если нет доступных цветов, используем новый цвет
                if not available_colors:
                    colors[v - 1] = k
                    k += 1
                else:
                    # Выбираем случайный доступный цвет
                    colors[v - 1] = random.choice(available_colors)

            return colors

        def dsatur_coloring(graph, nodes):
            """
            Раскраска графа с использованием алгоритма DSatur (Degree of Saturation).
            Приоритет отдается вершинам с наибольшей степенью насыщения (числом уникальных цветов соседей).
            :param graph: Граф (объект networkx.Graph), содержащий рёбра.
            :param nodes: Количество вершин в графе (нумерация с 1).
            :return: Список цветов (индексы от 0), где colors[i] — цвет вершины i+1.
            """
            # Создание списка смежности: для каждой вершины хранится множество её соседей
            adj_list = {i: set() for i in range(1, nodes + 1)}
            for u, v in graph.edges:
                adj_list[u].add(v)
                adj_list[v].add(u)
            # Инициализация списка цветов: -1 означает, что вершина не раскрашена
            colors = [-1] * nodes
            # Вычисление степеней вершин (число соседей)
            degrees = {v: len(adj_list[v]) for v in range(1, nodes + 1)}
            # Инициализация степени насыщения (число уникальных цветов соседей)
            saturation = {v: 0 for v in range(1, nodes + 1)}
            # Выбор первой вершины с максимальной степенью
            v = max(degrees, key=degrees.get)
            colors[v - 1] = 0  # Назначаем ей цвет 0
            colored = {v}  # Множество раскрашенных вершин
            # Продолжаем, пока не раскрасим все вершины
            while len(colored) < nodes:
                # Формируем список кандидатов: (степень насыщения, степень, вершина)
                candidates = [(saturation[v], degrees[v], v) for v in range(1, nodes + 1) if v not in colored]
                # Находим максимальную степень насыщения
                max_sat = max(candidates)[0]
                # Выбираем вершины с максимальной степенью насыщения
                max_sat_vertices = [v for sat, deg, v in candidates if sat == max_sat]
                # Случайный выбор среди вершин с максимальной степенью насыщения
                v = random.choice(max_sat_vertices)
                # Собираем цвета уже раскрашенных соседей
                used_colors = {colors[u - 1] for u in adj_list[v] if colors[u - 1] != -1}
                # Находим минимальный неконфликтный цвет
                color = 0
                while color in used_colors:
                    color += 1
                colors[v - 1] = color
                # Добавляем вершину в раскрашенные
                colored.add(v)
                # Обновляем степень насыщения для нераскрашенных соседей
                for u in adj_list[v]:
                    if u not in colored:
                        saturation[u] = len({colors[w - 1] for w in adj_list[u] if colors[w - 1] != -1})
            return colors

        start_time = time()

        # Выбор метода раскраски
        coloring_methods = {
            'Жадный': greedy_coloring,
            'Нез. множ.': independent_set_coloring,
            'Случайный': random_safe_coloring,
            'DSATUR': dsatur_coloring
        }
        if coloring_method not in coloring_methods:
            raise ValueError(
                "Недопустимый метод раскраски. Выберите: 'greedy', 'independent_set', 'random_safe', 'dsatur'")

        # Инициализация популяции без конфликтов
        population = []
        for _ in range(population_size):
            individual = coloring_methods[coloring_method](self.graph, self.nodes)
            population.append(individual)

        # Переменные для отслеживания минимального хроматического числа
        min_chromatic_number = float('inf')
        best_coloring = None
        min_chromatic_history = []  # История минимального хроматического числа

        # Списки для визуализации
        best_fitness_history = []
        avg_fitness_history = []
        worst_fitness_history = []
        chromatic_number_history = []
        best_individuals = []

        col_count = 0
        all_count = 0

        min_povt_count = 0

        if povt_count != 0:
            # Основной цикл
            while col_count < povt_count:
                # Оценка приспособленности и поиск минимального хроматического числа
                conflict_scores = []
                fitness_scores = []
                valid_colorings = []  # Корректные раскраски (без конфликтов)

                for individual in population:
                    conflicts = 0
                    for u, v in self.graph.edges:
                        if individual[u - 1] == individual[v - 1]:
                            conflicts += 1
                    num_colors = len(set(individual))
                    conflict_scores.append(conflicts)
                    fitness_scores.append(-conflicts - 1.0 * num_colors)

                    # Если раскраска корректна, проверяем число цветов
                    if conflicts == 0 and num_colors < min_chromatic_number:
                        min_chromatic_number = num_colors
                        best_coloring = individual.copy()

                    if conflicts == 0:
                        valid_colorings.append(num_colors)

                # print(f"Поколение {generation} {f'| Минимальная в поколении: {min(valid_colorings)}' if len(valid_colorings) > 0 else ''}")
                if all_count == 0:
                    start_chromatic_number = min(valid_colorings)
                    min_povt_count = start_chromatic_number

                all_count += 1

                if min_povt_count != min_chromatic_number:
                    min_povt_count = min_chromatic_number
                else:
                    col_count += 1

                # for i in range(len(population)):
                #     print(f"{i}. Конфликтов: {conflict_scores[i]} Особь: {population[i]} Кол-во цветов: {len(set(population[i]))}")
                #
                # print()

                # Сохранение статистики
                best_fitness = max(fitness_scores)
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                worst_fitness = min(fitness_scores)
                best_individual = population[fitness_scores.index(best_fitness)]
                chromatic_number = len(set(best_individual))

                best_fitness_history.append(best_fitness)
                avg_fitness_history.append(avg_fitness)
                worst_fitness_history.append(worst_fitness)
                chromatic_number_history.append(chromatic_number)
                best_individuals.append(best_individual.copy())

                # Сохраняем минимальное хроматическое число в этом поколении
                min_chromatic_history.append(min_chromatic_number if valid_colorings else chromatic_number)

                # Элитизм: сохраняем 10% лучших особей
                elite_count = max(1, population_size // 10)
                elite_indices = np.argsort(fitness_scores)[-elite_count:]
                new_population = [population[idx].copy() for idx in elite_indices]
                # print(f"Элитизм: {[population[idx].copy() for idx in elite_indices]}")
                # print()
                # print(f"Селекция:")
                # Селекция для оставшихся особей
                for b in range(population_size - elite_count):
                    candidates = random.sample(range(population_size), 2)
                    winner = candidates[0] if fitness_scores[candidates[0]] > fitness_scores[candidates[1]] else \
                    candidates[
                        1]
                    # print(f"{b}. особь №{candidates[0]}({fitness_scores[candidates[0]]}) > особь №{candidates[1]}({fitness_scores[candidates[1]]})") if fitness_scores[candidates[0]] > fitness_scores[candidates[1]] else print(f"{b}. особь №{candidates[0]}({fitness_scores[candidates[0]]}) < особь №{candidates[1]}({fitness_scores[candidates[1]]})")
                    new_population.append(population[winner].copy())
                # print()

                # Кроссовер
                new_population = perform_crossover(new_population, crossover_rate, self.graph)

                # print("Мутация:")
                # Мутация
                for i in range(population_size):
                    new_population[i] = mutate(new_population[i], self.graph, mutation_rate, i)
                # print()

                population = new_population
        else:
            # Основной цикл
            for generation in range(generations):
                # Оценка приспособленности и поиск минимального хроматического числа
                conflict_scores = []
                fitness_scores = []
                valid_colorings = []  # Корректные раскраски (без конфликтов)

                for individual in population:
                    conflicts = 0
                    for u, v in self.graph.edges:
                        if individual[u - 1] == individual[v - 1]:
                            conflicts += 1
                    num_colors = len(set(individual))
                    conflict_scores.append(conflicts)
                    fitness_scores.append(-conflicts - 1.0 * num_colors)

                    # Если раскраска корректна, проверяем число цветов
                    if conflicts == 0 and num_colors < min_chromatic_number:
                        min_chromatic_number = num_colors
                        best_coloring = individual.copy()

                    if conflicts == 0:
                        valid_colorings.append(num_colors)

                # print(f"Поколение {generation} {f'| Минимальная в поколении: {min(valid_colorings)}' if len(valid_colorings) > 0 else ''}")
                if generation == 0:
                    start_chromatic_number = min(valid_colorings)

                # for i in range(len(population)):
                #     print(f"{i}. Конфликтов: {conflict_scores[i]} Особь: {population[i]} Кол-во цветов: {len(set(population[i]))}")
                #
                # print()

                # Сохранение статистики
                best_fitness = max(fitness_scores)
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                worst_fitness = min(fitness_scores)
                best_individual = population[fitness_scores.index(best_fitness)]
                chromatic_number = len(set(best_individual))

                best_fitness_history.append(best_fitness)
                avg_fitness_history.append(avg_fitness)
                worst_fitness_history.append(worst_fitness)
                chromatic_number_history.append(chromatic_number)
                best_individuals.append(best_individual.copy())

                # Сохраняем минимальное хроматическое число в этом поколении
                min_chromatic_history.append(min_chromatic_number if valid_colorings else chromatic_number)

                # Элитизм: сохраняем 10% лучших особей
                elite_count = max(1, population_size // 10)
                elite_indices = np.argsort(fitness_scores)[-elite_count:]
                new_population = [population[idx].copy() for idx in elite_indices]
                # print(f"Элитизм: {[population[idx].copy() for idx in elite_indices]}")
                # print()
                # print(f"Селекция:")
                # Селекция для оставшихся особей
                for b in range(population_size - elite_count):
                    candidates = random.sample(range(population_size), 2)
                    winner = candidates[0] if fitness_scores[candidates[0]] > fitness_scores[candidates[1]] else candidates[
                        1]
                    # print(f"{b}. особь №{candidates[0]}({fitness_scores[candidates[0]]}) > особь №{candidates[1]}({fitness_scores[candidates[1]]})") if fitness_scores[candidates[0]] > fitness_scores[candidates[1]] else print(f"{b}. особь №{candidates[0]}({fitness_scores[candidates[0]]}) < особь №{candidates[1]}({fitness_scores[candidates[1]]})")
                    new_population.append(population[winner].copy())
                # print()

                # Кроссовер
                new_population = perform_crossover(new_population, crossover_rate, self.graph)

                # print("Мутация:")
                # Мутация
                for i in range(population_size):
                    new_population[i] = mutate(new_population[i], self.graph, mutation_rate, i)
                # print()

                population = new_population

        # Финальная проверка лучшей раскраски
        best_individual = max(population, key=fitness)

        self.list_colors = [self.sys_colors[color % len(self.sys_colors)] for color in best_coloring or best_individual]

        res_time = f"{time() - start_time:.4f}"

        # print(f"Приближенное хроматическое число: {min_chromatic_number} | Раскраска графа: {best_coloring}")

        with open('test_all_methods.txt', 'a') as file:
            dict_usual = {}
            dict_usual["method"] = f"Генетический"
            dict_usual["time"] = res_time
            dict_usual["nodes"] = self.nodes
            dict_usual["edges"] = self.edge_count
            dict_usual["result"] = min_chromatic_number
            file.write(str(dict_usual) + "\n")

        # Возвращаем минимальное хроматическое число и данные для визуализации
        return dict_usual

    def genetic_coloring(self, population_size=5, generations=5, mutation_rate=0.1, elite_size=5):
        """
        Генетический алгоритм для нахождения хроматического числа графа.
        """
        print("\n=== НАЧАЛО ГЕНЕТИЧЕСКОГО АЛГОРИТМА ===")
        print(f"Параметры алгоритма:")
        print(f"- Размер популяции: {population_size}")
        print(f"- Количество поколений: {generations}")
        print(f"- Вероятность мутации: {mutation_rate}")
        print(f"- Размер элиты: {elite_size}")

        start_time = time()
        print("\n1. Подготовка данных:")
        adjacency_list = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes}
        print(f"- Создан список смежности для {len(adjacency_list)} вершин")
        print(adjacency_list)

        def create_individual(max_colors):
            coloring = {}
            for node in self.graph.nodes:
                neighbor_colors = {coloring.get(neighbor) for neighbor in adjacency_list[node] if neighbor in coloring}
                available_colors = [c for c in range(max_colors) if c not in neighbor_colors]
                if available_colors:
                    coloring[node] = random.choice(available_colors)
                else:                    coloring[node] = random.randrange(max_colors)
            return coloring

        def fitness(coloring):
            conflicts = 0
            colors_used = set(coloring.values())
            for node in self.graph.nodes:
                for neighbor in adjacency_list[node]:
                    if coloring[node] == coloring[neighbor]:
                        conflicts += 1
            return -conflicts / 2, len(colors_used)

        def crossover(parent1, parent2):
            crossover_point = random.randint(1, self.nodes - 1)
            sorted_nodes = sorted(self.graph.nodes)
            child = {}
            for i, node in enumerate(sorted_nodes):
                if i < crossover_point:
                    child[node] = parent1[node]
                else:
                    child[node] = parent2[node]
            return child

        def mutate(coloring, max_colors):
            mutated = coloring.copy()
            for node in self.graph.nodes:
                if random.random() < mutation_rate:
                    neighbor_colors = {mutated.get(neighbor) for neighbor in adjacency_list[node] if
                                       neighbor in mutated}
                    available_colors = [c for c in range(max_colors) if c not in neighbor_colors]
                    if available_colors:
                        mutated[node] = random.choice(available_colors)
                    else:
                        mutated[node] = random.randrange(max_colors)
            return mutated

        print("\n2. Инициализация начальных параметров:")
        initial_colors = len(self.graph.nodes)
        current_best_colors = initial_colors
        print(f"- Начальное предположение о хроматическом числе: {initial_colors} цветов")

        population = [create_individual(initial_colors) for _ in range(population_size)]
        print(f"- Начальная популяция из {population_size} особей:")
        for i in population:
            print(i)

        print("\n3. Начало эволюционного процесса:")
        for generation in range(generations):
            print(f"\n=== ПОКОЛЕНИЕ {generation + 1}/{generations} ===")
            print("1. Оценка пригодности популяции:")

            # Оценка пригодности всех особей
            fitness_scores = []
            for idx, individual in enumerate(population):
                conflicts, colors_used = fitness(individual)
                fitness_scores.append((individual, (conflicts, colors_used)))
                print(f"  Особа #{idx + 1}: конфликты={-conflicts}, цветов={colors_used}, {individual}")

            # Сортировка по пригодности
            fitness_scores.sort(key=lambda x: (x[1][0], -x[1][1]), reverse=True)

            # Анализ лучших особей
            top_3 = fitness_scores[:3]
            best_coloring, (best_conflicts, best_colors_used) = top_3[0]
            avg_conflicts = sum(score[1][0] for score in fitness_scores) / len(fitness_scores)

            print("\n2. Анализ поколения:")
            print(f"- ЛУЧШАЯ ОСОБЬ: конфликты={-best_conflicts}, цветов={best_colors_used}")
            print(f"- Топ-3 особи:")
            for i, (ind, (conf, colors)) in enumerate(top_3):
                print(f"  {i + 1}. Конфликты={-conf}, Цветов={colors}")

            print(f"- Среднее количество конфликтов в популяции: {-avg_conflicts:.2f}")
            print(
                f"- Диапазон цветов в популяции: {min(f[1][1] for f in fitness_scores)}-{max(f[1][1] for f in fitness_scores)}")

            # Проверка на допустимую раскраску
            if best_conflicts == 0:
                unique_colors = set(best_coloring.values())
                num_colors = len(unique_colors)

                if num_colors < current_best_colors:
                    current_best_colors = num_colors
                    print("\n3. НАЙДЕНО УЛУЧШЕНИЕ:")
                    print(f"! ДОПУСТИМАЯ РАСКРАСКА С {num_colors} ЦВЕТАМИ !")
                    print(f"Цвета в раскраске: {sorted(unique_colors)}")

                    # Анализ распределения цветов
                    color_dist = {}
                    for color in best_coloring.values():
                        color_dist[color] = color_dist.get(color, 0) + 1
                    print("Распределение цветов:")
                    for color, count in sorted(color_dist.items()):
                        print(f"  Цвет {color}: {count} вершин ({count / self.nodes:.1%})")

                    if num_colors > 1:
                        print(f"\n4. УМЕНЬШЕНИЕ ЦВЕТОВ:")
                        new_max_colors = num_colors - 1
                        print(f"Пробуем уменьшить количество цветов до {new_max_colors}")
                        print("Создаем новую популяцию...")
                        population = [create_individual(new_max_colors) for _ in range(population_size)]
                        continue

            # Формирование нового поколения
            print("\n5. СОЗДАНИЕ НОВОГО ПОКОЛЕНИЯ:")
            next_generation = [coloring for coloring, _ in fitness_scores[:elite_size]]
            print(f"- Элитные особи (сохранено {elite_size} лучших):")
            for i, (ind, (conf, colors)) in enumerate(fitness_scores[:elite_size]):
                print(f"  #{i + 1}: конфликты={-conf}, цветов={colors}")

            print("\n6. ПРОЦЕСС РАЗМНОЖЕНИЯ:")
            children_count = 0
            while len(next_generation) < population_size:
                children_count += 1
                # Турнирный отбор
                tournament1 = random.sample(fitness_scores[:population_size // 2], 3)
                parent1 = max(tournament1, key=lambda x: x[1][0])[0]
                parent1_conf = -max(tournament1, key=lambda x: x[1][0])[1][0]
                parent1_colors = max(tournament1, key=lambda x: x[1][0])[1][1]

                tournament2 = random.sample(fitness_scores[:population_size // 2], 3)
                parent2 = max(tournament2, key=lambda x: x[1][0])[0]
                parent2_conf = -max(tournament2, key=lambda x: x[1][0])[1][0]
                parent2_colors = max(tournament2, key=lambda x: x[1][0])[1][1]

                print(f"\n  Потомок #{children_count}:")
                print(f"  Родитель 1: конфликты={parent1_conf}, цветов={parent1_colors}, {tournament1[1][0]}")
                print(f"  Родитель 2: конфликты={parent2_conf}, цветов={parent2_colors}, {tournament2[1][0]}")

                # Кроссовер
                child = crossover(parent1, parent2)
                child_conf, child_colors = fitness(child)
                print(f"  После кроссовера: конфликты={-child_conf}, цветов={child_colors}, {child}")

                # Мутация
                child = mutate(child, current_best_colors)
                child_conf, child_colors = fitness(child)
                print(f"  После мутации: конфликты={-child_conf}, цветов={child_colors}, {child}")

                next_generation.append(child)

            population = next_generation
            # print(f"\nИтоговый размер нового поколения: {len(population)} особей")

        print("\n4. Завершение алгоритма:")
        best_coloring = max(population, key=lambda x: fitness(x)[0])
        unique_colors = set(best_coloring.values())
        num_colors = len(unique_colors)

        color_map = {color: self.sys_colors[i % len(self.sys_colors)] for i, color in enumerate(unique_colors)}
        self.list_colors = [color_map[best_coloring[node]] for node in range(1, self.nodes + 1)]

        res_time = f"{time() - start_time:.4f}"
        print(f"- Наилучшая раскраска использует {num_colors} цветов")
        print(f"- Время выполнения: {res_time} секунд")

        with open('test_all_methods.txt', 'a') as file:
            dict_usual = {
                "method": "genetic",
                "time": res_time,
                "nodes": self.nodes,
                "edges": self.edge_count,
                "result": num_colors
            }
            file.write(str(dict_usual) + "\n")

        print("=== ГЕНЕТИЧЕСКИЙ АЛГОРИТМ ЗАВЕРШЕН ===\n")
        return num_colors

    def improved_genetic_coloring(self, population_size=100, generations=500, initial_mutation_rate=0.4, elite_size=30,
                                  local_search_prob=0.7, flag=True):
        """
        Улучшенный генетический алгоритм для нахождения хроматического числа графа.

        Параметры:
            population_size: Размер популяции
            generations: Максимальное количество поколений
            initial_mutation_rate: Начальная вероятность мутации
            elite_size: Количество лучших особей, сохраняемых в каждом поколении
            local_search_prob: Вероятность применения локального поиска

        Возвращает:
            Хроматическое число, найденное генетическим алгоритмом
        """
        import random

        # Начинаем отсчет времени
        start_time = time()

        # Получаем список смежности для более быстрой проверки соседей
        adjacency_list = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes}

        # Сортируем вершины по степени (для начальной популяции и локального поиска)
        sorted_by_degree = sorted(self.graph.nodes, key=lambda x: self.graph.degree[x], reverse=True)

        # Вычисляем максимальную степень для начальной оценки
        max_degree = max(dict(self.graph.degree).values())

        # Оценка нижней границы хроматического числа (размер максимальной клики)
        def estimate_max_clique():
            # Жадный алгоритм для поиска приближенной максимальной клики
            best_clique = []
            for start_node in sorted_by_degree:
                clique = [start_node]
                candidates = set(adjacency_list[start_node])

                while candidates:
                    # Выбираем вершину, соединенную со всеми вершинами в текущей клике
                    next_node = None
                    max_connections = -1

                    for node in candidates:
                        connections = sum(1 for c in clique if node in adjacency_list[c])
                        if connections == len(clique) and self.graph.degree[node] > max_connections:
                            next_node = node
                            max_connections = self.graph.degree[node]

                    if next_node:
                        clique.append(next_node)
                        candidates = candidates.intersection(set(adjacency_list[next_node]))
                    else:
                        break

                if len(clique) > len(best_clique):
                    best_clique = clique

            return len(best_clique)

        # Находим начальную оценку хроматического числа
        clique_size = estimate_max_clique()
        initial_colors = max(clique_size, (max_degree + 1) // 2 + 1)

        if flag == True:
            current_best_colors = min(initial_colors + 2, self.nodes)  # Начинаем с чуть большего значения
        else:
            current_best_colors = self.greedy_coloring()  # Жадный алгоритм

        # print(f"Начальная оценка хроматического числа: {initial_colors} (клика: {clique_size}, макс. степень: {max_degree})")
        # print()

        # Функция для создания начальной популяции с учетом жадной стратегии
        def create_initial_population(pop_size, max_colors):
            population = []
            for _ in range(pop_size):
                # Добавляем случайность в порядок вершин для разнообразия
                shuffled_nodes = sorted_by_degree.copy()
                # Перемешиваем частично, сохраняя некоторый порядок по степени
                for i in range(len(shuffled_nodes) - 1):
                    if random.random() < 0.3:  # 30% шанс обмена
                        j = min(i + random.randint(1, 3), len(shuffled_nodes) - 1)
                        shuffled_nodes[i], shuffled_nodes[j] = shuffled_nodes[j], shuffled_nodes[i]

                coloring = {}
                for node in shuffled_nodes:
                    # Получаем цвета соседей
                    neighbor_colors = {coloring.get(neighbor) for neighbor in adjacency_list[node] if
                                       neighbor in coloring}
                    # Находим доступные цвета
                    available_colors = [c for c in range(max_colors) if c not in neighbor_colors]
                    if available_colors:
                        # С некоторой вероятностью выбираем минимальный доступный цвет
                        if random.random() < 0.7:
                            coloring[node] = min(available_colors)
                        else:
                            coloring[node] = random.choice(available_colors)
                    else:
                        coloring[node] = random.randrange(max_colors)

                population.append(coloring)
            return population

        # Улучшенная функция пригодности с балансировкой целей
        def fitness(coloring, current_target_colors):
            # Подсчитываем конфликты (смежные вершины с одинаковым цветом)
            conflicts = 0
            colors_used = set(coloring.values())
            num_colors = len(colors_used)

            # Для инкрементальной оценки сначала проверим вершины с конфликтами
            conflict_nodes = set()

            for node in self.graph.nodes:
                node_conflicts = sum(1 for neighbor in adjacency_list[node]
                                     if neighbor in coloring and coloring[node] == coloring[neighbor])
                if node_conflicts > 0:
                    conflicts += node_conflicts
                    conflict_nodes.add(node)

            # Штраф за превышение целевого количества цветов
            color_penalty = max(0, num_colors - current_target_colors) * 5

            # Если нет конфликтов, увеличиваем вес минимизации цветов
            if conflicts == 0:
                color_penalty *= 2

            # Основная оценка: отрицательные конфликты (выше лучше) и штраф за цвета
            main_fitness = -conflicts / 2 - color_penalty

            # Вторичные метрики для дифференциации решений с одинаковым основным значением пригодности
            # Используем распределение цветов как вторичную метрику
            color_distribution = {}
            for color in coloring.values():
                color_distribution[color] = color_distribution.get(color, 0) + 1

            # Штраф за неравномерное распределение цветов (если нет конфликтов)
            distribution_penalty = 0
            if conflicts == 0 and num_colors > 0:
                avg_nodes_per_color = self.nodes / num_colors
                distribution_penalty = sum(
                    abs(count - avg_nodes_per_color) for count in color_distribution.values()) / self.nodes

            return main_fitness, -distribution_penalty, -num_colors, conflict_nodes

        # Улучшенный кроссовер с учетом конфликтов
        def improved_crossover(parent1, parent2, fit1, fit2):
            # Получаем множества конфликтных вершин для обоих родителей
            conflict_nodes1 = fit1[3]
            conflict_nodes2 = fit2[3]

            # Создаем базовый шаблон для ребенка
            # Берем цвета из лучшего родителя для неконфликтных вершин
            child = {}
            better_parent = parent1 if fit1[0] > fit2[0] else parent2
            worse_parent = parent2 if fit1[0] > fit2[0] else parent1

            # Для неконфликтных вершин берем цвета из лучшего родителя
            non_conflict_nodes = set(self.graph.nodes) - (conflict_nodes1.union(conflict_nodes2))
            for node in non_conflict_nodes:
                child[node] = better_parent[node]

            # Для конфликтных вершин выбираем цвет из любого родителя или комбинируем
            all_conflict_nodes = conflict_nodes1.union(conflict_nodes2)
            for node in all_conflict_nodes:
                # Если узел конфликтует только у одного родителя, берем цвет от другого
                if node in conflict_nodes1 and node not in conflict_nodes2:
                    child[node] = parent2[node]
                elif node in conflict_nodes2 and node not in conflict_nodes1:
                    child[node] = parent1[node]
                else:
                    # Если узел конфликтует у обоих, выбираем случайно или с учетом соседей
                    neighbor_colors1 = {parent1[n] for n in adjacency_list[node] if n in parent1}
                    neighbor_colors2 = {parent2[n] for n in adjacency_list[node] if n in parent2}

                    # Проверяем, есть ли конфликты с этими цветами
                    if parent1[node] not in neighbor_colors1:
                        child[node] = parent1[node]
                    elif parent2[node] not in neighbor_colors2:
                        child[node] = parent2[node]
                    else:
                        # Оба цвета конфликтуют, выбираем случайно
                        child[node] = random.choice([parent1[node], parent2[node]])

            return child

        # Адаптивная мутация с учетом структуры графа
        def adaptive_mutation(coloring, mutation_rate, target_colors):
            mutated = coloring.copy()
            all_colors = set(range(target_colors))

            # Находим конфликтные вершины
            conflict_nodes = []
            for node in self.graph.nodes:
                conflicts = [neighbor for neighbor in adjacency_list[node]
                             if neighbor in mutated and mutated[node] == mutated[neighbor]]
                if conflicts:
                    conflict_nodes.append((node, len(conflicts)))

            # Сортируем по количеству конфликтов (сначала больше)
            conflict_nodes.sort(key=lambda x: x[1], reverse=True)

            # Повышенный шанс мутации для конфликтных узлов
            for node, num_conflicts in conflict_nodes:
                # Адаптивная вероятность мутации, зависящая от количества конфликтов
                node_mutation_rate = min(0.9, mutation_rate * (1 + num_conflicts / 2))

                if random.random() < node_mutation_rate:
                    # Получаем цвета соседей
                    neighbor_colors = {mutated.get(neighbor) for neighbor in adjacency_list[node] if
                                       neighbor in mutated}
                    # Находим доступные цвета
                    available_colors = list(all_colors - neighbor_colors)

                    if available_colors:
                        mutated[node] = random.choice(available_colors)
                    else:
                        # Если нет доступных цветов, выбираем цвет, минимизирующий конфликты
                        color_conflicts = {}
                        for color in range(target_colors):
                            conflicts = sum(1 for neighbor in adjacency_list[node]
                                            if neighbor in mutated and mutated[neighbor] == color)
                            color_conflicts[color] = conflicts

                        # Выбираем цвет с минимальным количеством конфликтов
                        best_colors = [c for c, conf in color_conflicts.items()
                                       if conf == min(color_conflicts.values())]
                        mutated[node] = random.choice(best_colors)

            # Стандартная мутация для остальных вершин с более низкой вероятностью
            non_conflict_nodes = set(self.graph.nodes) - {node for node, _ in conflict_nodes}
            for node in non_conflict_nodes:
                if random.random() < mutation_rate / 2:  # Половина стандартной вероятности
                    neighbor_colors = {mutated.get(neighbor) for neighbor in adjacency_list[node] if
                                       neighbor in mutated}
                    available_colors = list(all_colors - neighbor_colors)

                    if available_colors:
                        mutated[node] = random.choice(available_colors)

            return mutated

        # Локальный поиск для улучшения решения
        def local_search(coloring, max_iterations=100):
            improved = coloring.copy()
            all_colors = set(improved.values())

            for _ in range(max_iterations):
                improvement = False

                # Находим конфликтные вершины
                conflict_nodes = []
                for node in self.graph.nodes:
                    conflicts = [neighbor for neighbor in adjacency_list[node]
                                 if neighbor in improved and improved[node] == improved[neighbor]]
                    if conflicts:
                        conflict_nodes.append((node, len(conflicts)))

                if not conflict_nodes:
                    # Пытаемся уменьшить количество цветов
                    if len(all_colors) > 1:
                        # Выбираем цвет, используемый меньше всего
                        color_counts = {}
                        for color in improved.values():
                            color_counts[color] = color_counts.get(color, 0) + 1

                        least_used_color = min(color_counts, key=color_counts.get)

                        # Пытаемся перекрасить вершины с этим цветом
                        nodes_with_color = [node for node, color in improved.items() if color == least_used_color]
                        remaining_colors = all_colors - {least_used_color}

                        for node in nodes_with_color:
                            neighbor_colors = {improved[neighbor] for neighbor in adjacency_list[node] if
                                               neighbor in improved}
                            available_colors = list(remaining_colors - neighbor_colors)

                            if available_colors:
                                improved[node] = random.choice(available_colors)
                                improvement = True

                        # Обновляем множество цветов
                        all_colors = set(improved.values())
                    break

                # Сортируем по количеству конфликтов
                conflict_nodes.sort(key=lambda x: x[1], reverse=True)

                # Пытаемся исправить конфликты
                for node, _ in conflict_nodes:
                    neighbor_colors = {improved[neighbor] for neighbor in adjacency_list[node] if neighbor in improved}
                    all_possible_colors = set(range(max(all_colors) + 2))  # +2 для возможности добавления нового цвета
                    available_colors = list(all_possible_colors - neighbor_colors)

                    if available_colors:
                        # Предпочитаем уже используемые цвета, если возможно
                        existing_colors = [c for c in available_colors if c in all_colors]
                        if existing_colors:
                            improved[node] = random.choice(existing_colors)
                        else:
                            improved[node] = min(available_colors)  # Выбираем минимальный новый цвет

                        improvement = True
                        all_colors = set(improved.values())
                        break  # Пересчитываем конфликты после каждого изменения

                if not improvement:
                    break

            return improved

        # Создаем начальную популяцию
        population = create_initial_population(population_size, current_best_colors)

        # Адаптивные параметры
        mutation_rate = initial_mutation_rate
        stagnation_counter = 0
        best_fitness_history = []

        # Эволюционный процесс
        best_chromatic_number = self.nodes  # Начинаем с худшего случая
        best_coloring = None

        for generation in range(generations):
            # Оцениваем пригодность
            fitness_scores = [(ind, fitness(ind, current_best_colors)) for ind in population]

            # Сортируем по пригодности (первый критерий - конфликты и штраф за цвета)
            fitness_scores.sort(key=lambda x: (x[1][0], x[1][1], x[1][2]), reverse=True)

            # Отслеживаем лучшую пригодность
            current_best_fitness = fitness_scores[0][1][0]
            best_fitness_history.append(current_best_fitness)

            # Проверка на стагнацию (если лучшая пригодность не улучшается)
            if len(best_fitness_history) > 10 and all(
                    best_fitness_history[-1] <= f for f in best_fitness_history[-10:]):
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            # Адаптация параметров при стагнации
            if stagnation_counter > 5:
                # Увеличиваем мутацию для выхода из локального оптимума
                mutation_rate = min(0.8, mutation_rate * 1.5)
                stagnation_counter = 0
                # print(f"Увеличена мутация до {mutation_rate:.2f} из-за стагнации")
            else:
                # Постепенно уменьшаем мутацию для лучшей сходимости
                mutation_rate = max(0.05, mutation_rate * 0.95)

            # Проверяем, есть ли у лучшей особи конфликты
            best_coloring_candidate, (best_fitness, _, _, _) = fitness_scores[0]
            unique_colors = set(best_coloring_candidate.values())
            num_colors = len(unique_colors)

            # Нашли раскраску без конфликтов
            if best_fitness >= 0:  # Нет конфликтов (и возможно небольшой штраф за цвета)
                # Применяем локальный поиск для дальнейшего улучшения
                improved_coloring = local_search(best_coloring_candidate)
                improved_colors = len(set(improved_coloring.values()))

                if improved_colors < num_colors:
                    best_coloring_candidate = improved_coloring
                    num_colors = improved_colors
                    # print(f"Лок. поиск: {improved_coloring}")
                    # print(f"Локальный поиск улучшил решение: {num_colors} цветов")

                if num_colors < best_chromatic_number:
                    best_chromatic_number = num_colors
                    best_coloring = best_coloring_candidate
                    # print(f"Допустимая раскраска: {best_coloring_candidate}")
                    # print(f"Поколение {generation}: Найдена допустимая раскраска с {num_colors} цветами")
                    # print()

                    # Сохраняем раскраску в графе
                    color_map = {color: self.sys_colors[i % len(self.sys_colors)] for i, color in
                                 enumerate(set(best_coloring.values()))}
                    self.list_colors = [color_map[best_coloring[node]] for node in range(1, self.nodes + 1)]

                    # Пытаемся создать новую популяцию с одним цветом меньше
                    if num_colors > clique_size:  # Не пытаемся опуститься ниже размера клики
                        current_best_colors = num_colors - 1
                        # Частично обновляем популяцию
                        new_individuals = create_initial_population(population_size // 2, current_best_colors)
                        population = population[:population_size // 2] + new_individuals
                        continue

            # Применяем элитную стратегию
            next_generation = [ind for ind, _ in fitness_scores[:elite_size]]

            # Применяем локальный поиск к некоторым элитным особям
            for i in range(min(3, elite_size)):
                if random.random() < local_search_prob:
                    next_generation[i] = local_search(next_generation[i])

            # Заполняем остаток популяции
            while len(next_generation) < population_size:
                # Турнирный отбор с динамическим размером турнира
                tournament_size = 3 + stagnation_counter // 3  # Увеличиваем размер турнира при стагнации
                tournament_size = min(tournament_size, len(fitness_scores) // 3)

                # Выбираем родителей
                tournament1 = random.sample(fitness_scores[:population_size // 2 + 5], tournament_size)
                parent1, fit1 = max(tournament1, key=lambda x: x[1][0])

                tournament2 = random.sample(fitness_scores[:population_size // 2 + 5], tournament_size)
                parent2, fit2 = max(tournament2, key=lambda x: x[1][0])

                # Улучшенный кроссовер
                child = improved_crossover(parent1, parent2, fit1, fit2)

                # Адаптивная мутация
                child = adaptive_mutation(child, mutation_rate, current_best_colors)

                # Локальный поиск с некоторой вероятностью
                if random.random() < local_search_prob:
                    child = local_search(child)

                next_generation.append(child)

            population = next_generation

            # # Отображаем прогресс
            # if generation % 10 == 0 or generation == generations - 1:
            #     print(
            #         f"Поколение {generation}: Текущая цель = {current_best_colors}, Лучший результат = {best_chromatic_number}")

        # Если не нашли допустимую раскраску, используем лучшую найденную
        if best_coloring is None:
            best_coloring = max(population, key=lambda x: fitness(x, self.nodes)[0])
            best_chromatic_number = len(set(best_coloring.values()))

            # Сохраняем раскраску в графе
            color_map = {color: self.sys_colors[i % len(self.sys_colors)] for i, color in
                         enumerate(set(best_coloring.values()))}
            self.list_colors = [color_map[best_coloring[node]] for node in range(1, self.nodes + 1)]

        # Выводим информацию о времени
        res_time = f"{time() - start_time:.4f}"
        # print(f"Улучшенный генетический алгоритм завершен за {res_time} секунд")
        # print(f"Найденное хроматическое число: {best_chromatic_number}")

        if flag == True:
            with open('test_all_methods.txt', 'a') as file:
                dict_usual = {}
                dict_usual["method"] = "MAGU"
                dict_usual["time"] = res_time
                dict_usual["nodes"] = self.nodes
                dict_usual["edges"] = self.edge_count
                dict_usual["result"] = best_chromatic_number
                dict_usual["current_best_colors"] = "True"
                file.write(str(dict_usual)+"\n")
        else:
            with open('test_all_methods.txt', 'a') as file:
                dict_usual = {}
                dict_usual["method"] = "MAGU"
                dict_usual["time"] = res_time
                dict_usual["nodes"] = self.nodes
                dict_usual["edges"] = self.edge_count
                dict_usual["result"] = best_chromatic_number
                dict_usual["current_best_colors"] = "False"
                file.write(str(dict_usual)+"\n")

        return dict_usual


if __name__ == '__main__':
    """ Метод МАГУ """
    # g = Graph(4, 5)
    # chromatic_number = g.method_MAGU()
    # print(f"Хроматическое число МАГУ графа: {chromatic_number}")
    # print(f"Исправленный МАГУ графа: {chromatic_number-1}")
    # # g.draw_graph()
    #
    # """ Жадный алгоритм """
    # # g = Graph(20)
    # print("Приближенное хроматическое число:", g.greedy_coloring()+1)
    # # g.draw_graph()

    # """ Генетический алгоритм """
    # g = Graph(16, 120)
    # print("Приближенное хроматическое число (генетический алгоритм):", g.genetic_algorithm_coloring())
    # g.draw_graph()

    # """ Генетический алгоритм v. 2"""
    # g = Graph(6, 7)
    # print("Приближенное хроматическое число (генетический алгоритм v. 2):", g.genetic_coloring())
    # g.draw_graph()

    # """ Генетический алгоритм v. 3"""
    # g = Graph(20)
    # print("Приближенное хроматическое число (генетический алгоритм v. 3):", g.improved_genetic_coloring())
    # g.draw_graph()

    # for j in range(20, 25):
    #     for i in range(10):
    #         g = Graph(9, j)
    #         g.method_MAGU()
    #         g.greedy_coloring()
    #         g.genetic_coloring()
    #         g.improved_genetic_coloring()
    #         g.improved_genetic_coloring(flag=False)
    #         print(i)

    for i in range(50):
        if i < 50:
            g = Graph(16)
        elif 50 <= i < 100:
            g = Graph(16)
        else:
            g = Graph(17)
        # g.greedy_coloring()
        # g.genetic_algorithm_coloring(coloring_method="greedy")
        # g.genetic_algorithm_coloring(coloring_method="independent_set")
        g.genetic_algorithm_coloring()
        # g.genetic_algorithm_coloring(coloring_method="dsatur")
        # g.improved_genetic_coloring()
        print(i)
