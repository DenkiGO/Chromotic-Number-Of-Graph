import re
import numpy as np
from itertools import product
import sympy as sp


def create_incidence_matrix(graph):
    nodes = sorted(set(node for edge in graph for node in edge))  # Получаем уникальные вершины
    node_index = {node: i for i, node in enumerate(nodes)}  # Создаём соответствие вершина -> индекс

    incidence_matrix = np.zeros((len(nodes), len(graph)), dtype=int)

    for edge_idx, (u, v) in enumerate(graph):
        incidence_matrix[node_index[u]][edge_idx] = 1
        incidence_matrix[node_index[v]][edge_idx] = 1

    return incidence_matrix, node_index


def get_dnf(incidence_matrix, node_index):
    dnf_terms = []
    for edge_idx in range(incidence_matrix.shape[1]):
        nodes = [node for node, idx in node_index.items() if incidence_matrix[idx][edge_idx] == 1]
        dnf_terms.append(f'({" + ".join(["x" + str(n) for n in nodes])})')
    return " * ".join(dnf_terms)


def expand_dnf(dnf_expression):
    terms = dnf_expression.split(" * ")
    expanded_terms = [term.strip("() ").split(" + ") for term in terms]
    expanded_products = ["".join(sorted(set(prod))) for prod in product(*expanded_terms)]
    return " + ".join(sorted(set(expanded_products)))


def distribute_terms(terms):
    if len(terms) == 1:
        return terms[0]
    first, rest = terms[0], distribute_terms(terms[1:])
    return sorted(set("".join(sorted(set(a + b))) for a in first for b in rest))


def expand_brackets(dnf_expression):
    terms = [term.strip("() ").split(" + ") for term in dnf_expression.split(" * ")]
    expanded_terms = distribute_terms(terms)
    return " + ".join(expanded_terms)


def add_missing_nodes(conjunction, all_nodes):
    # Извлекаем только числа из строки, игнорируя 'x'
    existing_nodes = set(re.findall(r'\d+', conjunction))  # Используем регулярные выражения для извлечения чисел
    result = set(existing_nodes.pop())
    missing_nodes = all_nodes - result
    return "".join(sorted(missing_nodes))


# Задаем граф
graph = [(1, 2), (1, 3), (1, 8), (2, 3), (2, 4), (2, 5), (3, 5), (5, 6), (5, 7), (6, 7), (6, 8), (7, 8)]
incidence_matrix, node_index = create_incidence_matrix(graph)

# Получаем ДНФ выражение
dnf_expression = get_dnf(incidence_matrix, node_index)

# Переводим выражение в раскрытые скобки
expanded_dnf = expand_brackets(dnf_expression)

# Для каждой конъюнкции из шагов (x3x4x8, x4x5x8, ...) добавляем недостающие вершины
all_nodes = set(str(i) for i in node_index.keys())

conjunctions = expanded_dnf.split(" + ")

# Добавляем недостающие вершины только для тех, у которых есть недостающие вершины
conjunctions_with_missing_nodes = [add_missing_nodes(conjunction, all_nodes) for conjunction in conjunctions if add_missing_nodes(conjunction, all_nodes)]

sorted_strings_desc = sorted(conjunctions_with_missing_nodes, key=len, reverse=True)

all_nudes_final = {}

num_colors = 0

while len(sorted_strings_desc) != 0:
    all_nodes = set(sorted_strings_desc[0])
    num_colors += 1

    filtered_strings = [
        ''.join([digit for digit in s if digit not in all_nodes]) for s in sorted_strings_desc
    ]

    filtered_strings = [s for s in filtered_strings if s]

    sorted_strings_desc = sorted(filtered_strings, key=len, reverse=True)


print(f"Матрица инцидентности: \n{incidence_matrix}\n")
print(f"ДНФ: {dnf_expression}\n")
# Раскрываем скобки
expanded_expression = sp.expand(dnf_expression)
# Выводим результат
print(expanded_expression)
print(f"Хроматическое число: {num_colors}")
