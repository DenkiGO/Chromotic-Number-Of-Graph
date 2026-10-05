import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ttkthemes import ThemedTk

import threading

from main import Graph


class App(ThemedTk):
    def __init__(self):
        super().__init__(theme="alt")

        self.processing_thread = None
        self.stop_processing = False

        self.title("Анализатор графов")
        self.geometry("1000x630")
        self.minsize(1000, 630)

        self.style = ttk.Style(self)
        self.style.configure('TLabel', padding=5)
        self.style.configure('TButton', padding=5)
        self.style.configure('TEntry', padding=5)

        main_frame = ttk.Frame(self, borderwidth=5, relief="raised", padding=7)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top_left_frame = ttk.Frame(left_frame, borderwidth=5, relief="sunken")
        top_left_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=top_left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        self.console = tk.Text(left_frame, height=10, wrap=tk.WORD, font=('Consolas', 10), borderwidth=5, relief="sunken")
        self.console.pack(fill=tk.BOTH, expand=False)
        self.console.configure(state='disabled')

        right_frame = ttk.Frame(main_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Создаем Notebook (вкладки)
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка 1 (Основные настройки)
        self.tab1 = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.tab1, text="Тестирование")

        # Вкладка 1 (Основные настройки)
        self.tab2 = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.tab2, text="Анализ")

        # Добавляем вкладки
        self.setup_tab1()
        self.add_tab2()

    def setup_tab1(self):
        """Настройка содержимого первой вкладки"""
        # Фрейм управления
        control_frame = ttk.LabelFrame(self.tab1, text="Управление", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Фрейм алгоритмов
        alg_frame = ttk.LabelFrame(self.tab1, text="Алгоритмы", padding=10)
        alg_frame.pack(fill=tk.X, pady=(0, 10))

        # Фрейм генетического алгоритма (изначально скрыт)
        self.genetic_frame = ttk.LabelFrame(self.tab1, text="Параметры ген. алг.", padding=10)

        # Элементы управления
        ttk.Label(control_frame, text="Количество вершин:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.input_nodes = ttk.Entry(control_frame)
        self.input_nodes.insert(0, "5")
        self.input_nodes.grid(row=0, column=1, sticky=tk.EW, pady=2)

        ttk.Label(control_frame, text="Количество графов:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.input_graphs = ttk.Entry(control_frame)
        self.input_graphs.insert(0, "1")
        self.input_graphs.grid(row=1, column=1, sticky=tk.EW, pady=2)

        ttk.Label(control_frame, text="Вероятность рёбер:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.input_edges = ttk.Entry(control_frame)
        self.input_edges.insert(0, "0.6")
        self.input_edges.grid(row=2, column=1, sticky=tk.EW, pady=2)

        self.random_edges_var = tk.BooleanVar(value=True)
        self.random_edges_check = ttk.Checkbutton(
            control_frame, text="Случайные рёбра", variable=self.random_edges_var, command=self.toggle_edges_input)
        self.random_edges_check.grid(row=3, columnspan=2, sticky=tk.W, pady=5)

        self.plot_btn = ttk.Button(control_frame, text="Построить и раскрасить", command=self.start_processing)
        self.plot_btn.grid(row=5, columnspan=2, sticky=tk.EW, pady=2)

        self.clear_btn = ttk.Button(control_frame, text="Очистить консоль", command=self.clear_console)
        self.clear_btn.grid(row=6, columnspan=2, sticky=tk.EW, pady=2)

        # Элементы алгоритмов
        self.greedy_edges_var = tk.BooleanVar(value=True)
        self.greedy_edges_check = ttk.Checkbutton(
            alg_frame, text="Жадный", variable=self.greedy_edges_var)
        self.greedy_edges_check.grid(row=0, column=0, sticky=tk.EW, padx=5)

        self.genetic_edges_var = tk.BooleanVar(value=False)
        self.genetic_edges_check = ttk.Checkbutton(
            alg_frame, text="Генетический", variable=self.genetic_edges_var, command=self.gen_alg_input)
        self.genetic_edges_check.grid(row=0, column=1, sticky=tk.EW, padx=5)

        self.MAGU_edges_var = tk.BooleanVar(value=False)
        self.MAGU_edges_check = ttk.Checkbutton(
            alg_frame, text="Магу", variable=self.MAGU_edges_var)
        self.MAGU_edges_check.grid(row=0, column=2, sticky=tk.EW, padx=5)

        # Элементы генетического алгоритма
        ttk.Label(self.genetic_frame, text="Кол-во особей:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.input_population = ttk.Entry(self.genetic_frame)
        self.input_population.insert(0, "50")
        self.input_population.grid(row=0, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self.genetic_frame, text="Вер. мутации:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.input_mutation = ttk.Entry(self.genetic_frame)
        self.input_mutation.insert(0, "0.5")
        self.input_mutation.grid(row=1, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self.genetic_frame, text="Вер. кроссовера:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.input_crossover = ttk.Entry(self.genetic_frame)
        self.input_crossover.insert(0, "0.5")
        self.input_crossover.grid(row=2, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self.genetic_frame, text="Нач. популяция:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.population_var = tk.StringVar(value="Жадный")
        self.population_combo = ttk.Combobox(
            self.genetic_frame,
            textvariable=self.population_var,
            values=["Жадный", "Случайный", "DSATUR", "Нез. множ."],
            state="readonly"
        )
        self.population_combo.grid(row=3, column=1, sticky=tk.EW, pady=2)

        ttk.Label(self.genetic_frame, text="Кол-во повторов:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.input_povt = ttk.Entry(self.genetic_frame)
        self.input_povt.insert(0, "50")
        self.input_povt.grid(row=4, column=1, sticky=tk.EW, pady=2)

        self.povt_edges_var = tk.BooleanVar(value=True)
        self.povt_edges_check = ttk.Checkbutton(
            self.genetic_frame,
            text="Повтор значения",
            variable=self.povt_edges_var,
            command=self.povt_chrom_input
        )
        self.povt_edges_check.grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5)

        control_frame.columnconfigure(1, weight=1)
        alg_frame.columnconfigure(1, weight=1)
        self.genetic_frame.columnconfigure(1, weight=1)

        # Изначально скрываем фрейм генетического алгоритма
        self.genetic_frame.pack_forget()

    def add_tab2(self):
        """Вторая вкладка"""

        # Получаем путь к директории проекта
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        control_frame = ttk.LabelFrame(self.tab2, text="Управление", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.postr_btn = ttk.Button(control_frame, text="Выбор файла", command=self.select_file)
        self.postr_btn.grid(row=0, column=0, sticky=tk.EW, pady=2)

        self.input_file = ttk.Entry(control_frame)
        self.input_file.insert(0, "")
        self.input_file.grid(row=0, column=1, sticky=tk.EW, pady=2, ipadx=17)
        self.input_file.configure(state="readonly")  # Запрет редактирования

        ttk.Label(control_frame, text="Метрика:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.srav_var = tk.StringVar(value="Время")
        self.srav_combo = ttk.Combobox(
            control_frame,
            textvariable=self.srav_var,
            values=["Время", "Точность"],
            state="readonly"
        )
        self.srav_combo.grid(row=1, column=1, sticky=tk.EW, pady=2)

        self.build_btn = ttk.Button(control_frame, text="Построить", command=self.postr_result)
        self.build_btn.grid(row=2, columnspan=2, sticky=tk.EW, pady=2)

    def rename_test_file(self):
        original_file = "test_all_methods.txt"

        # Проверяем, существует ли исходный файл
        if not os.path.exists(original_file):
            print(f"Файл {original_file} не найден")
            return

        # Ищем доступное имя для нового файла
        counter = 1
        while True:
            new_name = f"test{counter}.txt"
            if not os.path.exists(new_name):
                try:
                    os.rename(original_file, new_name)
                    self.log(f"Результат сохранен в {new_name}")
                    break
                except Exception as e:
                    print(f"Ошибка при переименовании: {e}")
                    break
            counter += 1

    def process_graphs_threaded(self):
        """Метод для выполнения в отдельном потоке"""
        try:
            self.ax.clear()
            self.canvas.draw_idle()
            self.count_nodes = int(self.input_nodes.get())
            self.count_graph = int(self.input_graphs.get())
            self.count_edges = self.input_edges.get()

            all_count = 0
            all_count_methods = 0

            # Подсчет общего количества операций
            if self.MAGU_edges_var.get():
                all_count_methods += 1
            if self.genetic_edges_var.get():
                all_count_methods += 1
            if self.greedy_edges_var.get():
                all_count_methods += 1

            all_count_methods = all_count_methods * self.count_graph

            for i in range(self.count_graph):
                if self.stop_processing:
                    break

                # Создаем граф
                if self.random_edges_var.get():
                    g = Graph(self.count_nodes, edges_procent=float(self.count_edges))
                else:
                    g = Graph(self.count_nodes, edges=int(self.count_edges))

                # Обработка графа жадным алгоритмом
                if self.greedy_edges_var.get() and not self.stop_processing:
                    g.greedy_coloring()
                    all_count += 1
                    self.update_progress(all_count, all_count_methods)

                # Обработка генетическим алгоритмом
                if self.genetic_edges_var.get() and not self.stop_processing:
                    pop = int(self.input_population.get())
                    mut = float(self.input_mutation.get())
                    cros = float(self.input_crossover.get())
                    povt = int(self.input_povt.get())
                    nach_pop = self.population_var.get()

                    if self.povt_edges_var.get():
                        g.genetic_algorithm_coloring(population_size=pop, mutation_rate=mut,
                                                     crossover_rate=cros, coloring_method=nach_pop,
                                                     povt_count=povt)
                    else:
                        g.genetic_algorithm_coloring(population_size=pop, generations=povt,
                                                     mutation_rate=mut, crossover_rate=cros,
                                                     coloring_method=nach_pop)

                    all_count += 1
                    self.update_progress(all_count, all_count_methods)

                # Обработка MAGU
                if self.MAGU_edges_var.get() and not self.stop_processing:
                    g.improved_genetic_coloring()
                    all_count += 1
                    self.update_progress(all_count, all_count_methods)

        except Exception as e:
            self.log(f"Ошибка в потоке обработки: {str(e)}")
        finally:
            self.processing_thread = None

    def update_progress(self, current, total):
        """Обновление прогресса"""
        progress = (current / total) * 100
        self.clear_console()
        self.log(f"---------------------------------------- {int(progress)}% / 100% ---------------------------------------")
        if progress == 100:
            self.rename_test_file()

    def continue_threaded(self):
        self.plot_btn.config(state="disabled")
        self.clear_btn.config(state="disabled")
        if self.count_graph == 1:
            if self.random_edges_var.get():
                g = Graph(self.count_nodes, edges_procent=float(self.count_edges))
            else:
                g = Graph(self.count_nodes, edges=int(self.count_edges))
            if self.greedy_edges_var.get():
                res = g.greedy_coloring()
                g.draw_graph(ax=self.ax)
                self.canvas.draw()
                self.log(f"Алгоритм: {res['method']}\nВремя: {res['time']}\nКол-во вершин: {res['nodes']}\nКол-во рёбер: {res['edges']}\nПриближенное хроматическое число: {res['result']}\n")
            if self.genetic_edges_var.get():
                pop = int(self.input_population.get())
                mut = float(self.input_mutation.get())
                cros = float(self.input_crossover.get())
                povt = int(self.input_povt.get())
                nach_pop = self.population_var.get()
                if self.povt_edges_var.get():
                    res = g.genetic_algorithm_coloring(population_size=pop, mutation_rate=mut, crossover_rate=cros,
                                   coloring_method=nach_pop, povt_count=povt)
                else:
                    res = g.genetic_algorithm_coloring(population_size=pop, generations=povt, mutation_rate=mut,
                                                 crossover_rate=cros, coloring_method=nach_pop)
                g.draw_graph(ax=self.ax)
                self.canvas.draw()
                self.log(
                    f"Алгоритм: {res['method']}\nВремя: {res['time']}\nКол-во вершин: {res['nodes']}\nКол-во рёбер: {res['edges']}\nПриближенное хроматическое число: {res['result']}\n")
            if self.MAGU_edges_var.get():
                res = g.improved_genetic_coloring()
                g.draw_graph(ax=self.ax)
                self.canvas.draw()
                self.log(
                    f"Алгоритм: {res['method']}\nВремя: {res['time']}\nКол-во вершин: {res['nodes']}\nКол-во рёбер: {res['edges']}\nХроматическое число: {res['result']}\n")
        elif self.count_graph > 1:
            self.process_graphs_threaded()
        self.plot_btn.config(state="normal")
        self.clear_btn.config(state="normal")

    def start_processing(self):
        """Запуск обработки в отдельном потоке"""
        self.ax.clear()
        self.canvas.draw_idle()
        self.count_nodes = int(self.input_nodes.get())
        self.count_graph = int(self.input_graphs.get())
        self.count_edges = self.input_edges.get()
        if self.processing_thread and self.processing_thread.is_alive():
            self.log("Обработка уже выполняется!")
            return

        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self.continue_threaded)
        self.processing_thread.daemon = True  # Поток завершится при закрытии программы
        self.processing_thread.start()

    def postr_result(self):
        self.ax.clear()
        self.canvas.draw_idle()
        if self.srav_var.get() == "Время":
            self.build_time_comparison()
        else:
            self.build_comparison()


    def build_time_comparison(self):
        """Построение графика сравнения среднего времени выполнения методов из файла"""
        try:
            import json
            from collections import defaultdict

            # Очистка предыдущего содержимого осей
            self.ax.clear()

            # Чтение данных из файла
            with open(f'{self.filee}', 'r') as file:
                lines = file.readlines()

            # Сбор данных
            data = defaultdict(list)

            for line in lines:
                try:
                    entry = json.loads(line.replace("'", '"'))
                    method = entry['method']
                    time = float(entry['time'])  # Предполагается, что время в файле
                    data[method].append(time)
                except (json.JSONDecodeError, KeyError):
                    continue

            # Вычисление среднего времени
            avg_times = {method: sum(times) / len(times)
                         for method, times in data.items()}

            # Построение диаграммы
            methods = list(avg_times.keys())
            avg_values = list(avg_times.values())

            self.ax.bar(methods, avg_values, color=['blue', 'green', 'red', 'purple'])
            self.ax.set_xlabel('Метод')
            self.ax.set_ylabel('Среднее время (сек)')
            self.ax.set_title('Сравнение среднего времени выполнения методов')
            self.ax.grid(True, axis='y')

            # Добавление подписи
            subtitle = ", ".join([f"{method}: {avg_times[method]:.2f} сек"
                                  for method in methods])
            self.log(f"Сред. результат: {subtitle}")

            # Обновление канваса
            self.canvas.draw()

        except Exception as e:
            self.log(f"Ошибка при построении графика времени: {str(e)}")

    def build_comparison(self):
        """Построение графика сравнения методов из файла"""
        try:
            import json
            from collections import defaultdict

            # Очистка предыдущего содержимого осей
            self.ax.clear()

            # Чтение данных из файла
            with open(f'{self.filee}', 'r') as file:
                lines = file.readlines()

            # Сбор данных
            data = defaultdict(list)

            for line in lines:
                try:
                    entry = json.loads(line.replace("'", '"'))
                    method = entry['method']
                    result = int(entry['result'])
                    data[method].append(result)
                except json.JSONDecodeError:
                    continue

            # Вычисление среднего результата
            avg_results = {method: sum(results) / len(results)
                           for method, results in data.items()}

            # Построение диаграммы
            methods = list(avg_results.keys())
            avg_values = list(avg_results.values())

            self.ax.bar(methods, avg_values, color=['blue', 'green', 'red', 'purple'])
            self.ax.set_xlabel('Метод')
            self.ax.set_ylabel('Средний результат')
            self.ax.set_title('Сравнение средних результатов по методам')
            self.ax.grid(True, axis='y')

            # Добавление подписи
            subtitle = ", ".join([f"{method}: {avg_results[method]:.2f}"
                                  for method in methods])
            self.log(f"Сред. результат: {subtitle}")

            # Обновление канваса
            self.canvas.draw()

        except Exception as e:
            self.log(f"Ошибка при построении графика: {str(e)}")

    def select_file(self):
        """Открывает диалог выбора файла в директории проекта"""
        self.filepath = filedialog.askopenfilename(
            title="Выберите файл",
            initialdir=self.project_dir,  # Указываем начальную директорию
            filetypes=(("Текстовые файлы", "*.txt"), ("Все файлы", "*.*"))
        )

        if self.filepath:  # Если файл выбран (не нажали "Отмена")
            self.filee = self.filepath
            self.input_file.configure(state="normal")
            self.input_file.delete(0, tk.END)
            self.input_file.insert(0, self.filepath.split('/')[-1])
            self.input_file.configure(state="readonly")

    def about(self):
        """Окно 'О программе'"""
        about_text = "Анализатор графов\nВерсия 1.0\n\nПриложение для визуализации и раскраски графов"
        messagebox.showinfo("О программе", about_text)

    def quit_app(self):
        """Выход из приложения"""
        if messagebox.askokcancel("Выход", "Вы уверены, что хотите выйти?"):
            self.destroy()

    def toggle_edges_input(self):
        """Переключение метки поля ввода рёбер в зависимости от чекбокса"""
        label = self.nametowidget(self.input_edges.winfo_parent()).grid_slaves(row=2, column=0)[0]
        label.configure(text="Вероятность рёбер:" if self.random_edges_var.get() else "Количество рёбер:")

    def povt_chrom_input(self):
        """Переключение метки поля ввода повтора"""
        label = self.nametowidget(self.input_povt.winfo_parent()).grid_slaves(row=4, column=0)[0]
        label.configure(text="Кол-во повторов:" if self.povt_edges_var.get() else "Кол-во поколений:")

    def gen_alg_input(self):
        """Переключение метки поля ввода ген. алгоритма"""
        if self.genetic_edges_var.get():
            self.genetic_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.genetic_frame.pack_forget()

    def log(self, message):
        """Добавление сообщения в консоль"""
        self.console.configure(state='normal')
        self.console.insert(tk.END, message + "\n")
        self.console.configure(state='disabled')
        self.console.see(tk.END)

    def clear_console(self):
        """Очистка консоли"""
        self.console.configure(state='normal')
        self.console.delete(1.0, tk.END)
        self.console.configure(state='disabled')

if __name__ == "__main__":
    app = App()
    app.mainloop()