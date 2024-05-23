import numpy as np

# Початкові параметри
theta1 = 5
theta0 = 7
learning_rate = 0.1
precision = 0.25

# Навчальна пара
x1 = 1
y1 = 1

# Функція гіпотези
def hypothesis(x, theta1, theta0):
    return x * theta1 + theta0

# Функція вартості (середньоквадратична помилка)
def cost_function(x, y, theta1, theta0):
    return (hypothesis(x, theta1, theta0) - y) ** 2 / 2

# Градієнтний спуск
def gradient_descent(x, y, theta1, theta0, learning_rate, precision):
    cost = cost_function(x, y, theta1, theta0)
    iterations = 0
    
    while cost > precision:
        temp1 = theta1 - learning_rate * (hypothesis(x, theta1, theta0) - y) * x
        temp0 = theta0 - learning_rate * (hypothesis(x, theta1, theta0) - y)
        theta1 = temp1
        theta0 = temp0
        cost = cost_function(x, y, theta1, theta0)
        iterations += 1
        print(f"Iteration {iterations}: theta1 = {theta1}, theta0 = {theta0}, cost = {cost}")
    
    return theta1, theta0

theta1, theta0 = gradient_descent(x1, y1, theta1, theta0, learning_rate, precision)
print(f"Оптимізовані параметри: theta1 = {theta1}, theta0 = {theta0}")
