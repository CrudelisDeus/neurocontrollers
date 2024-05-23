import tensorflow as tf
import numpy as np

# 1. Створення випадкових даних
np.random.seed(42)
x_data = np.random.rand(1000, 1).astype(np.float32)
y_data = (2 * x_data + 1 + np.random.normal(0, 2, (1000, 1))).astype(np.float32)

# Функція для створення датасету
def create_dataset(x_data, y_data):
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    return dataset.shuffle(buffer_size=1000).batch(100).repeat()

# 2. Створення датасету
dataset = create_dataset(x_data, y_data)
iterator = iter(dataset)

# 3. Ініціалізація змінних для моделі
k = tf.Variable(tf.random.normal([1]), name='k', dtype=tf.float32)
b = tf.Variable(tf.zeros([1]), name='b', dtype=tf.float32)

# 4. Визначення функції моделі
def model(X):
    return k * X + b

# 5. Визначення функції втрат
def loss_fn(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true - y_pred))

# 6. Визначення оптимізатора
optimizer = tf.optimizers.SGD(learning_rate=0.001)

# 7. Виконання тренування моделі
for epoch in range(20000):
    for _ in range(10):  # Виконання 10 кроків навчання на кожній епосі
        x_batch, y_batch = next(iterator)
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = loss_fn(y_batch, y_pred)
        gradients = tape.gradient(loss, [k, b])
        optimizer.apply_gradients(zip(gradients, [k, b]))
    
    if (epoch + 1) % 100 == 0:
        print(f"Епоха {epoch + 1}: {loss.numpy():.6f}, k={k.numpy()[0]:.4f}, b={b.numpy()[0]:.4f}")
