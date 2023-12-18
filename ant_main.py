import numpy as np
from sklearn.preprocessing import LabelEncoder
from tkinter import *
from simpful import *

def distance(point1, point2, graph):
    return graph[point1][point2]

def ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q, graph):
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf

    for iteration in range(n_iterations):
        paths = []
        path_lengths = []

        for ant in range(n_ants):
            visited = [False] * n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0

            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))

                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = pheromone[current_point, unvisited_point] ** alpha / distance(
                        current_point, unvisited_point, graph) ** beta

                probabilities /= np.sum(probabilities)

                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(current_point, next_point, graph)
                visited[next_point] = True
                current_point = next_point

            paths.append(path)
            path_lengths.append(path_length)

            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        pheromone *= evaporation_rate

        for path, path_length in zip(paths, path_lengths):
            if path_length > 0:  # Добавляем проверку на ненулевую длину пути
                for i in range(n_points - 1):
                     pheromone[path[i], path[i + 1]] += Q / path_length
                pheromone[path[-1], path[0]] += Q / path_length

    return best_path

def fuzzy_rules_and_system_create():
    FS = FuzzySystem()

    S_1 = FuzzySet(function=Triangular_MF(a=5, b=15, c=35), term="quiet")
    S_2 = FuzzySet(function=Triangular_MF(a=35, b=40, c=45), term="medium")
    S_3 = FuzzySet(function=Triangular_MF(a=45, b=50, c=60), term="loud")
    FS.add_linguistic_variable("Sound", LinguisticVariable([S_1, S_2, S_3], concept="Laptop sound",
                                                             universe_of_discourse=[5, 60]))
    S_11 = FuzzySet(function=Triangular_MF(a=30, b=30, c=50), term="low")
    S_22 = FuzzySet(function=Triangular_MF(a=55, b=60, c=70), term="medium")
    S_33 = FuzzySet(function=Triangular_MF(a=70, b=85, c=100), term="high")
    FS.add_linguistic_variable("Temperature", LinguisticVariable([S_11, S_22, S_33], concept="Laptop temperature",
                                                             universe_of_discourse=[30, 100]))

    S_111 = FuzzySet(function=Triangular_MF(a=0, b=0, c=2), term="young")
    S_222 = FuzzySet(function=Triangular_MF(a=1, b=5, c=6), term="medium")
    S_333 = FuzzySet(function=Triangular_MF(a=4, b=7, c=10), term="old")
    FS.add_linguistic_variable("Age", LinguisticVariable([S_111, S_222, S_333], concept="Laptop age",
                                                             universe_of_discourse=[0, 10]))

    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=20), term="small")
    T_2 = FuzzySet(function=Triangular_MF(a=10, b=30, c=40), term="average")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=30, b=40, c=55, d=100), term="high")
    FS.add_linguistic_variable("Hardware_Failure", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 100]))

    R1 = "IF (Temperature IS high) THEN (Hardware_Failure IS high)"
    R2 = "IF (Temperature IS low) AND (Sound IS quiet) OR (Age IS young) THEN (Hardware_Failure IS small)"
    R3 = "IF (Temperature IS medium) OR (Age IS young) AND (Sound IS medium) THEN (Hardware_Failure IS average)"
    R4 = "IF (Temperature IS medium) OR (Age IS young) AND (Sound IS loud) THEN (Hardware_Failure IS high)"
    R5 = "IF (Temperature IS medium) AND (Age IS old) AND (Sound IS medium) THEN (Hardware_Failure IS average)"
    R6 = "IF (Temperature IS low) AND (Age IS medium) AND (Sound IS quiet) THEN (Hardware_Failure IS small)"
    R7 = "IF (Temperature IS low) AND (Age IS old) AND (Sound IS loud) THEN (Hardware_Failure IS high)"
    FS.add_rules([R1, R2, R3, R4, R5, R6, R7])

    # Пример задания значений переменных
    FS.set_variable("Sound",59)
    FS.set_variable("Temperature", 99)
    FS.set_variable("Age", 19)

    # Получение результата нечеткой системы
    #result = FS.Mamdani_inference(["Hardware_Failure"])
    #print(result)
    return FS

# def use_fuzzy_system(self, args):
#     FS = FuzzySystem()
#     text_vars = ["Sound", "Temperature", "Age"]
#     result_vars = ["Hardware_Failure"]
#     for text, arg in zip(text_vars, args):
#         FS.set_variable(text, arg)
#     results = FS.Mamdani_inference(result_vars)
#     # Возвращаем точки для алгоритма муравьиной колонии
#     return results["Hardware_Failure"]

def use_fuzzy_system(FS):
    result_vars = ["Hardware_Failure"]
    results = FS.Mamdani_inference(result_vars)
    # Возвращаем значения для алгоритма муравьиной колонии в виде списка
    return [results[var] for var in result_vars]

def use_ant_colony(result):
    # Используем результат нечеткой системы для оптимизации муравьиной колонией
    points = [int(value) for value in result]
    n_ants = 5  # Задайте необходимое количество муравьев
    n_iterations = 100  # Задайте необходимое количество итераций
    alpha = 1.0  # Задайте параметр alpha
    beta = 2.0  # Задайте параметр beta
    evaporation_rate = 0.5  # Задайте коэффициент испарения феромонов
    Q = 1.0  # Задайте количество феромонов, добавляемых каждым муравьем

    # Ваш граф
    graph = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])

    # Запускаем алгоритм муравьиной колонии
    
    optimal_path = ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q, graph)

    # if not optimal_path:
    #     print("No hardware issues detected.")
    #     return

    # Определение аппаратных неисправностей на основе оптимального пути
    detected_issues = set()

    for i in range(len(optimal_path) - 1):
        from_point = points_for_ant_colony[optimal_path[i]]
        to_point = points_for_ant_colony[optimal_path[i + 1]]

        if from_point == 'Hardware_Failure':
            detected_issues.add(to_point)
        elif to_point == 'Hardware_Failure':
            detected_issues.add(from_point)

    # Вывод обнаруженных аппаратных неисправностей
    if detected_issues:
        print("Detected hardware issues:", ", ".join(detected_issues))
    else:
        print("No hardware issues detected.")

# Вызываем функции
fuzzy_system = fuzzy_rules_and_system_create()
points_for_ant_colony = use_fuzzy_system(fuzzy_system)
use_ant_colony(points_for_ant_colony)
