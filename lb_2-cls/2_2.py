import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score

# Завантаження даних
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, names=columns)

# Попередня обробка даних
data = data.replace(' ?', np.nan).dropna()

# Кодування категоріальних змінних
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = preprocessing.LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Вибір ознак та міток
X = data.drop('income', axis=1).values
y = data['income'].values

# Зменшення розміру даних для швидкої оцінки
data_sample = data.sample(frac=0.1, random_state=5)
X_sample = data_sample.drop('income', axis=1).values
y_sample = data_sample['income'].values

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=5)

# Поліноміальне ядро
classifier_poly = SVC(kernel='poly', degree=8, random_state=0)
classifier_poly.fit(X_train, y_train)
f1_poly = cross_val_score(classifier_poly, X_sample, y_sample, scoring='f1_weighted', cv=3, n_jobs=-1).mean()

# Гаусове ядро
classifier_rbf = SVC(kernel='rbf', random_state=0)
classifier_rbf.fit(X_train, y_train)
f1_rbf = cross_val_score(classifier_rbf, X_sample, y_sample, scoring='f1_weighted', cv=3, n_jobs=-1).mean()

# Сигмоїдальне ядро
classifier_sigmoid = SVC(kernel='sigmoid', random_state=0)
classifier_sigmoid.fit(X_train, y_train)
f1_sigmoid = cross_val_score(classifier_sigmoid, X_sample, y_sample, scoring='f1_weighted', cv=3, n_jobs=-1).mean()

print(f"F1 score with Polynomial kernel: {round(100 * f1_poly, 2)}%")
print(f"F1 score with RBF kernel: {round(100 * f1_rbf, 2)}%")
print(f"F1 score with Sigmoid kernel: {round(100 * f1_sigmoid, 2)}%")
