import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Чтение данных из файла
with open('test_all_methods.txt', 'r') as file:
    lines = file.readlines()

# Сбор данных
data = defaultdict(list)

for line in lines:
    try:
        entry = json.loads(line.replace("'", '"'))  # Заменяем одинарные кавычки на двойные для корректного JSON
        method = entry['method']
        result = int(entry['result'])
        data[method].append(result)  # Сохраняем результаты для каждого метода
    except json.JSONDecodeError:
        # Пропускаем строки, которые не могут быть преобразованы в JSON
        continue

# Вычисление среднего результата для каждого метода
avg_results = {method: sum(results) / len(results) for method, results in data.items()}

# Построение диаграммы
methods = list(avg_results.keys())
avg_values = list(avg_results.values())

plt.figure(figsize=(8, 6))
plt.bar(methods, avg_values, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Метод')
plt.ylabel('Средний результат (result)')
plt.title('Сравнение средних результатов по методам')
plt.grid(True, axis='y')  # Сетка только по оси Y

# Добавляем подпись с средними значениями
subtitle = ", ".join([f"{method}: {avg_results[method]:.2f}" for method in methods])
plt.figtext(0.5, 0.01, f"Средние значения: {subtitle}", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

plt.show()