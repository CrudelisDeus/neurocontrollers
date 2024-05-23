import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Дані
X = np.array([6, 7, 8, 9, 10, 12]).reshape(-1, 1)
Y = np.array([2, 3, 3, 4, 6, 5])

# Модель лінійної регресії
model = LinearRegression()
model.fit(X, Y)

# Прогнозовані значення
Y_pred = model.predict(X)

# Коефіцієнти
intercept = model.intercept_
slope = model.coef_[0]

print(f"Коефіцієнт β0 (intercept): {intercept}")
print(f"Коефіцієнт β1 (slope): {slope}")

plt.scatter(X, Y, color='blue', label='Експериментальні точки')
plt.plot(X, Y_pred, color='red', label='Лінія регресії')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Лінійна регресія')
plt.legend()
plt.grid(True)
plt.savefig('result.png')
