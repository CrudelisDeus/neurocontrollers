import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
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

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Створення та навчання класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)

# Прогнозування та оцінка моделі
y_test_pred = classifier.predict(X_test)
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3).mean()

print(f"F1 score: {round(100 * f1, 2)}%")
