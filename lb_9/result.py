import matplotlib
matplotlib.use('Agg')
import random
import numpy as np
import matplotlib.pyplot as plt

# Кількість точок та максимальне значення координат (порядковий номер студента)
N = 20

# Генеруємо випадкові координати точок
x1 = [random.uniform(0, N) for _ in range(N)]
x2 = [random.uniform(0, N) for _ in range(N)]

# Визначення вирішальних функцій
def g1(x1, x2):
    return 4*x1 + 4*x2 - 44

def g2(x1, x2):
    return 2*x1 + x2 - 24

def determine_class(x1, x2):
    if g1(x1, x2) < 0 and g2(x1, x2) >= 0:
        return 1
    elif g1(x1, x2) >= 0 and g2(x1, x2) >= 0:
        return 2
    elif g1(x1, x2) >= 0 and g2(x1, x2) < 0:
        return 3
    else:
        return 4

# Визначаємо клас для кожної точки
classes = [determine_class(x1[i], x2[i]) for i in range(N)]

# Виведемо результати класифікації
for i in range(N):
    print(f"Точка {i+1}: x1 = {x1[i]:.2f}, x2 = {x2[i]:.2f}, Клас = {classes[i]}")

# Налаштування кольорів для кожного класу
colors = ['red', 'green', 'blue', 'orange']

# Візуалізація точок
for i in range(N):
    plt.scatter(x1[i], x2[i], color=colors[classes[i]-1])

# Візуалізація вирішальних прямих
x_range = np.linspace(0, N, 100)
plt.plot(x_range, (44 - 4*x_range) / 4, label='g1: 4x1 + 4x2 = 44')
plt.plot(x_range, (24 - 2*x_range) / 1, label='g2: 2x1 + x2 = 24')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Класифікація точок на площині')
plt.legend()
plt.grid(True)
plt.savefig('result.png')
