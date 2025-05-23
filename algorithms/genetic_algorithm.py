import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import symengine as sp
import re
from time import time
from multiprocessing import Pool


class Graph:
    def __init__(self, nodes, edges=None):
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
        self.generate_random_edges()

    def generate_random_edges(self):
        """Метод создания случайных рёбер с указанным количеством"""
        if self.edges is None:
            self.edge_count = 0  # Инициализация счётчика рёбер
            # Если количество рёбер не указано, генерируем рёбра случайно
            for i in range(1, self.nodes + 1):
                for j in range(i + 1, self.nodes + 1):
                    if random.random() > 0.6:
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

    def draw_graph(self):
        """Визуализация графа"""
        if len(self.list_colors) != 0:
            pos = nx.circular_layout(self.graph)  # Или другой алгоритм/ручное задание координат
            nx.draw(self.graph, pos, with_labels=True, node_color=self.list_colors, edge_color='gray',
                    font_weight='bold')
            plt.show()
        else:
            pos = nx.circular_layout(self.graph)  # Или другой алгоритм/ручное задание координат
            nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_weight='bold')
            plt.show()

    def get_edges(self):
        """Получение списка ребер"""
        return list(self.graph.edges)

    @staticmethod
    def generate_graph_set(num_graphs, num_nodes, edges=None):
        """Создание и получение списка длинной num_graphs графов с количеством вершин num_nodes"""
        return {Graph(num_nodes, edges) for _ in range(num_graphs)}

    def genetic_algorithm_coloring(self, population_size=5, generations=5, mutation_rate=0.3, crossover_rate=0.5,
                                   coloring_method='random_safe'):
        """
        Генетический алгоритм с отслеживанием минимального хроматического числа во всех поколениях.
        :param population_size: Размер популяции.
        :param generations: Количество поколений.
        :param mutation_rate: Вероятность мутации.
        :param crossover_rate: Вероятность кроссовера.
        :param coloring_method: Метод раскраски ('greedy', 'independent_set', 'random_safe', 'dsatur').
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

            print("Кроссовер:")

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
                    print('---')
                    print(f"особь №{i}. Родитель 1: {parent1} ({fitness_parent1}), Родитель 2: {parent2} | точка кроссовера: {crossover_point}\nРебенок 1: {child1} ({fitness_child1}), Ребенок 2: {child2} ({fitness_child2}) | Лучший: {best_child} ({max(fitness_child1, fitness_child2, fitness_parent1)})")

                    # Заменяем только если потомок лучше родителя
                    if best_fitness > fitness_parent1:
                        new_population[i] = best_child

                    # Заменяем первого родителя (текущую особь) лучшим потомком
                    new_population[i] = best_child
            print()
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
                print("---")
                print(f"особь №{count_osob}. Мутация в вершине: {idx}. Новый цвет: {color}\n До {ind} | После {individual}")

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

        # Выбор метода раскраски
        coloring_methods = {
            'greedy': greedy_coloring,
            'independent_set': independent_set_coloring,
            'random_safe': random_safe_coloring,
            'dsatur': dsatur_coloring
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

            print(f"Поколение {generation} {f'| Минимальная в поколении: {min(valid_colorings)}' if len(valid_colorings) > 0 else ''}")
            if generation == 0:
                start_chromatic_number = min(valid_colorings)

            for i in range(len(population)):
                print(f"{i}. Конфликтов: {conflict_scores[i]} Особь: {population[i]} Кол-во цветов: {len(set(population[i]))}")

            print()

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
            print(f"Элитизм: {[population[idx].copy() for idx in elite_indices]}")
            print()
            print(f"Селекция:")
            # Селекция для оставшихся особей
            for b in range(population_size - elite_count):
                candidates = random.sample(range(population_size), 2)
                winner = candidates[0] if fitness_scores[candidates[0]] > fitness_scores[candidates[1]] else candidates[
                    1]
                print(f"{b}. особь №{candidates[0]}({fitness_scores[candidates[0]]}) > особь №{candidates[1]}({fitness_scores[candidates[1]]})") if fitness_scores[candidates[0]] > fitness_scores[candidates[1]] else print(f"{b}. особь №{candidates[0]}({fitness_scores[candidates[0]]}) < особь №{candidates[1]}({fitness_scores[candidates[1]]})")
                new_population.append(population[winner].copy())
            print()

            # Кроссовер
            new_population = perform_crossover(new_population, crossover_rate, self.graph)

            print("Мутация:")
            # Мутация
            for i in range(population_size):
                new_population[i] = mutate(new_population[i], self.graph, mutation_rate, i)
            print()

            population = new_population

        # Финальная проверка лучшей раскраски
        best_individual = max(population, key=fitness)

        self.list_colors = [self.sys_colors[color % len(self.sys_colors)] for color in best_coloring or best_individual]
        final_chromatic_number = len(set(best_coloring or best_individual))

        print(f"Приближенное хроматическое число: {min_chromatic_number} | Раскраска графа: {best_coloring}")

        # Возвращаем минимальное хроматическое число и данные для визуализации
        return min_chromatic_number, {
            'best_fitness': best_fitness_history,
            'avg_fitness': avg_fitness_history,
            'worst_fitness': worst_fitness_history,
            'chromatic_number': chromatic_number_history,
            'min_chromatic_number': min_chromatic_history,
            'best_individuals': best_individuals,
            'best_coloring': best_coloring or best_individual,
            'start_chromatic_number': start_chromatic_number
        }


if __name__ == '__main__':
    """ Генетический алгоритм """
    g = Graph(10)
    g.draw_graph()

    result = g.genetic_algorithm_coloring()

    print("Приближенное хроматическое число (генетический алгоритм):", result[0], result[1]['start_chromatic_number'])
    g.draw_graph()
