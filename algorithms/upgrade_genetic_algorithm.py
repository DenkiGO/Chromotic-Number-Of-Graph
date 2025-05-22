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

    def incidence_matrix(self):
        """Создание и получение матрицы инцидентности графа"""
        nodes_list = list(self.graph.nodes)
        edges_list = list(self.graph.edges)
        matrix = np.zeros((len(nodes_list), len(edges_list)), dtype=int)

        for edge_index, (u, v) in enumerate(edges_list):
            matrix[nodes_list.index(u)][edge_index] = 1
            matrix[nodes_list.index(v)][edge_index] = 1

        return matrix, {node: i for i, node in enumerate(nodes_list)}

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
            dict_usual["method"] = "greedy"
            dict_usual["time"] = res_time
            dict_usual["nodes"] = self.nodes
            dict_usual["edges"] = self.edge_count
            dict_usual["result"] = chromatic_number
            file.write(str(dict_usual)+"\n")

        return chromatic_number

    def improved_genetic_coloring(self, population_size=5, generations=5, initial_mutation_rate=0.2, crossover_rate=0.8, elite_size=3,
                                  local_search_prob=0.4, flag=True):
        """
        Улучшенный генетический алгоритм для нахождения хроматического числа графа.

        Параметры:
            population_size: Размер популяции
            generations: Максимальное количество поколений
            initial_mutation_rate: Начальная вероятность мутации
            elite_size: Количество лучших особей, сохраняемых в каждом поколении
            crossover_rate: Вероятность кроссовера
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

            if random.random() < crossover_rate:
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
                            chosen = random.choice([parent1[node], parent2[node]])
                            child[node] = chosen

                crossover_list['1'].append(parent1)
                crossover_list['2'].append(parent2)
                crossover_list['3'].append(child)

            else:
                child = better_parent

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

            flag_mutation = False

            node_list = []
            new_color_list = []

            # Стандартная мутация для остальных вершин с более низкой вероятностью
            non_conflict_nodes = set(self.graph.nodes) - {node for node, _ in conflict_nodes}
            for node in non_conflict_nodes:
                if random.random() < mutation_rate / 2:  # Половина стандартной вероятности
                    flag_mutation = True
                    mutation_list['1'].append(coloring)
                    node_list.append(node)

                    neighbor_colors = {mutated.get(neighbor) for neighbor in adjacency_list[node] if
                                       neighbor in mutated}
                    available_colors = list(all_colors - neighbor_colors)

                    if available_colors:
                        new_color = random.choice(available_colors)
                        mutated[node] = new_color
                        new_color_list.append(f"Изменение: {coloring[node]} → {new_color}")
            if flag_mutation:
                mutation_list['2'].append(mutated)
                mutation_list['3'].append(node_list)
                mutation_list['4'].append(new_color_list)

            return mutated

        # Локальный поиск для улучшения решения
        def local_search(coloring, max_iterations=100):
            improved = coloring.copy()
            all_colors = set(improved.values())

            flag_local_search = False

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
                else:
                    flag_local_search = True

            if flag_local_search:
                local_search_list['1'].append(coloring)
                local_search_list['2'].append(improved)

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

            min_chromatic_list = []

            for j in fitness_scores:
                min_chromatic_list.append(j[1][2])

            print(f"----- {generation} ПОКОЛЕНИЕ ----- | Минимальное приближенное хроматическое число: {str(max(min_chromatic_list))[1:]}")
            for i in range(len(population)):
                print(f"{i} Особь: {population[i]} | Кол-во цветов: {str(fitness_scores[i][1][2])[1:]}")
            print()

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

            print(f"Элитизм: {[ind for ind, _ in fitness_scores[:elite_size]]}")
            print()

            # Применяем локальный поиск к некоторым элитным особям
            for i in range(min(3, elite_size)):
                if random.random() < local_search_prob:
                    next_generation[i] = local_search(next_generation[i])

            # Инициализация счетчика турниров
            tournament_counter = 0

            print("Селекция:")

            count = 0

            crossover_list = {"1": [], "2": [], "3": []}

            mutation_list = {'1': [], '2': [], '3': [], '4': []}

            local_search_list = {'1': [], '2': [], '3': [], '4': []}

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

                # Получаем множества конфликтных вершин для обоих родителей
                conflict_nodes1 = fit1[3]
                conflict_nodes2 = fit2[3]

                better_parent = parent1 if fit1[0] > fit2[0] else parent2

                print(f"{count}. {parent1} ({fit1[0]}) {'>' if fit1[0] > fit2[0] else '<'} {parent2} ({fit2[0]})")
                count += 1

                # Улучшенный кроссовер
                child = improved_crossover(parent1, parent2, fit1, fit2)

                # Адаптивная мутация
                child = adaptive_mutation(child, mutation_rate, current_best_colors)

                # Локальный поиск с некоторой вероятностью
                if random.random() < local_search_prob:
                    child = local_search(child)

                next_generation.append(child)
            print()

            if len(crossover_list['1']) != 0:
                print("Кроссовер:")

            for i in range(len(crossover_list['1'])):
                print(f"{i}. Родитель 1: {crossover_list['1'][i]} | Родитель 2: {crossover_list['1'][i]}\nРебенок: {crossover_list['1'][i]}")

            print()

            if len(mutation_list['1']) != 0:
                print("Мутация:")

            for i in range(len(mutation_list['2'])):
                print(f"{i}. Исходная раскраска: {mutation_list['1'][i]}")
                for j in range(len(mutation_list['3'][i])):
                    print(f"Вершина: {mutation_list['3'][i][j]} | {mutation_list['4'][i][j]}")
                print(f"Итоговая раскраска: {mutation_list['2'][i]}")

            print()

            if len(local_search_list['1']) != 0:
                print("Локальный поиск:")

            for i in range(len(local_search_list['1'])):
                print(f"{i}. Начальная раскраска:: {local_search_list['1'][i]} | Итоговая раскраска: {local_search_list['2'][i]}")


            print()
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

        if flag == True:
            with open('test_all_methods.txt', 'a') as file:
                dict_usual = {}
                dict_usual["method"] = "improved_genetic"
                dict_usual["time"] = res_time
                dict_usual["nodes"] = self.nodes
                dict_usual["edges"] = self.edge_count
                dict_usual["result"] = best_chromatic_number
                dict_usual["current_best_colors"] = "True"
                file.write(str(dict_usual)+"\n")
        else:
            with open('test_all_methods.txt', 'a') as file:
                dict_usual = {}
                dict_usual["method"] = "improved_genetic"
                dict_usual["time"] = res_time
                dict_usual["nodes"] = self.nodes
                dict_usual["edges"] = self.edge_count
                dict_usual["result"] = best_chromatic_number
                dict_usual["current_best_colors"] = "False"
                file.write(str(dict_usual)+"\n")

        return best_chromatic_number


if __name__ == '__main__':
    """ Генетический алгоритм v. 3"""
    g = Graph(6, 8)
    g.draw_graph()
    print("Приближенное хроматическое число (генетический алгоритм v. 3):", g.improved_genetic_coloring())
    g.draw_graph()
