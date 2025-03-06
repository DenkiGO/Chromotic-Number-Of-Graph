import numpy as np
import networkx as nx
from numba import jit


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.sys_colors = ["#FF0000", "#00FFFF", "#FFFF00", "#0000FF", "#900020", "#808000", "#800080", "#008000"]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(1, nodes + 1))
        self.generate_random_edges()

    def generate_random_edges(self):
        edges = []
        for i in range(1, self.nodes + 1):
            for j in range(i + 1, self.nodes + 1):
                if np.random.random() > 0.6:
                    edges.append((i, j))
        self.graph.add_edges_from(edges)

    def incidence_matrix(self):
        nodes_list = list(self.graph.nodes)
        edges_list = list(self.graph.edges)
        matrix = np.zeros((len(nodes_list), len(edges_list)), dtype=np.int32)

        for edge_index, (u, v) in enumerate(edges_list):
            matrix[nodes_list.index(u), edge_index] = 1
            matrix[nodes_list.index(v), edge_index] = 1

        return matrix

    @staticmethod
    @jit(nopython=True)
    def process_incidence_matrix(incidence_matrix):
        rows, cols = incidence_matrix.shape
        dnf_terms = []

        for edge_idx in range(cols):
            nodes = []
            for node_idx in range(rows):
                if incidence_matrix[node_idx, edge_idx] == 1:
                    nodes.append(node_idx + 1)
            dnf_terms.append(np.array(nodes, dtype=np.int64))

        return dnf_terms

    @staticmethod
    @jit(nopython=True)
    def find_color_sets(dnf_terms, num_nodes):
        # Находим максимальную длину через явный цикл
        max_len = 0
        for term in dnf_terms:
            if len(term) > max_len:
                max_len = len(term)

        # Создаем 2D массив
        dnf_array = np.full((len(dnf_terms), max_len), -1, dtype=np.int64)
        for i in range(len(dnf_terms)):
            term = dnf_terms[i]
            for j in range(len(term)):
                dnf_array[i, j] = term[j]

        # Сортировка
        lengths = np.zeros(len(dnf_terms), dtype=np.int64)
        for i in range(len(dnf_terms)):
            cnt = 0
            for j in range(max_len):
                if dnf_array[i, j] != -1:
                    cnt += 1
            lengths[i] = cnt

        sorted_indices = np.argsort(-lengths)

        # Обработка цветов
        used = np.zeros(num_nodes + 1, dtype=np.bool_)  # +1 для индексации с 1
        color_sets = []

        for idx in sorted_indices:
            current_nodes = []
            for j in range(max_len):
                node = dnf_array[idx, j]
                if node == -1:
                    break
                if not used[node]:
                    current_nodes.append(node)

            if len(current_nodes) > 0:
                color_sets.append(np.array(current_nodes, dtype=np.int64))
                for node in current_nodes:
                    used[node] = True

        return color_sets

    def method_MAGU(self):
        incidence_matrix = self.incidence_matrix()
        dnf_terms = self.process_incidence_matrix(incidence_matrix)
        color_sets = self.find_color_sets(dnf_terms, self.nodes)

        color_map = {}
        for color_idx, nodes in enumerate(color_sets):
            for node in nodes:
                color_map[node] = self.sys_colors[color_idx % len(self.sys_colors)]

        self.list_colors = [
            color_map.get(i, self.sys_colors[0])
            for i in range(1, self.nodes + 1)
        ]
        return color_sets


# Тестирование
if __name__ == "__main__":
    import time

    start = time.time()
    g = Graph(100)
    print("Хроматическое число:", g.method_MAGU())
    print(f"Время: {time.time() - start:.2f} сек")