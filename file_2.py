import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Чтение данных из файла
with open('test_all_methods.txt', 'r') as file:
    lines = file.readlines()

# Сбор данных
experiment_data = defaultdict(list)
methods_order = []  # Для сохранения порядка методов

for i, line in enumerate(lines):
    try:
        entry = json.loads(line.replace("'", '"'))
        method = entry['method']
        result = int(entry['result'])

        # Сохраняем порядок методов при первом появлении
        if method not in methods_order:
            methods_order.append(method)

        experiment_data[method].append((i + 1, result))  # (номер эксперимента, результат)
    except json.JSONDecodeError:
        continue

# Построение графика
plt.figure(figsize=(12, 6))

print(experiment_data)

# Для каждого метода строим свою линию
for method in methods_order:
    if method in experiment_data:
        experiments = [x[0] for x in experiment_data[method]]
        results = [x[1] for x in experiment_data[method]]
        plt.plot(experiments, results, marker='o', label=method, linewidth=2)

plt.xlabel('Номер эксперимента')
plt.ylabel('Результат (result)')
plt.title('Динамика результатов по экспериментам')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()