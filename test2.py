def greedy_coloring(graph):
    n = len(graph)
    colors = [-1] * n

    for v in range(n):
        used_colors = set()

        # Check neighbors and mark their colors as used
        for neighbor in graph[v]:
            if colors[neighbor] != -1:
                used_colors.add(colors[neighbor])

        # Find the smallest available color
        color = 1
        while True:
            if color not in used_colors:
                colors[v] = color
                break
            color += 1

    # Find the maximum color used (chromatic number)
    chromatic_number = max(colors) + 1
    return chromatic_number

# Sample graph represented as an adjacency list
graph = [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]

# Find and output the chromatic number
chromatic_number = greedy_coloring(graph)
print("Chromatic Number:", chromatic_number)
