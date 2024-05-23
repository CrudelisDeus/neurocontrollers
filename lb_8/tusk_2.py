import numpy as np

# Навчальні дані
data = np.array([
    [1, 1],
    [1.2, 1.5],
    [2, 3.45],
    [2.8, 4.0123],
    [3.1, 3.09]
])

# Функція гіпотези
def hypothesis(x, theta1, theta0):
    return x * theta1 + theta0

# Функція вартості для всіх даних
def cost_function_all(data, theta1, theta0):
    total_cost = 0
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        total_cost += (hypothesis(x, theta1, theta0) - y) ** 2
    return total_cost / (2 * len(data))

# Градієнтний спуск для всіх даних
def gradient_descent_all(data, theta1, theta0, learning_rate, precision):
    cost = cost_function_all(data, theta1, theta0)
    iterations = 0
    
    while cost > precision:
        sum1 = 0
        sum0 = 0
        for i in range(len(data)):
            x = data[i, 0]
            y = data[i, 1]
            sum1 += (hypothesis(x, theta1, theta0) - y) * x
            sum0 += (hypothesis(x, theta1, theta0) - y)
        theta1 -= learning_rate * sum1 / len(data)
        theta0 -= learning_rate * sum0 / len(data)
        cost = cost_function_all(data, theta1, theta0)
        iterations += 1
        print(f"Iteration {iterations}: theta1 = {theta1}, theta0 = {theta0}, cost = {cost}")
    
    return theta1, theta0

# Початкові параметри
theta1 = 5
theta0 = 7
learning_rate = 0.1
precision = 0.25

# Виконання градієнтного спуску для всіх даних
theta1, theta0 = gradient_descent_all(data, theta1, theta0, learning_rate, precision)
print(f"Оптимізовані параметри: theta1 = {theta1}, theta0 = {theta0}")
