import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

# Завантаження даних
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, names=columns)

# Попередня обробка даних
data = data.replace(' ?', np.nan).dropna()

# Кодування категоріальних змінних
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Вибір ознак та міток
X = data.drop('income', axis=1).values
y = data['income'].values

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Створення моделей для income_data
models_income = []
models_income.append(('LR', OneVsRestClassifier(LogisticRegression(solver='liblinear'))))
models_income.append(('LDA', LinearDiscriminantAnalysis()))
models_income.append(('KNN', KNeighborsClassifier()))
models_income.append(('CART', DecisionTreeClassifier()))
models_income.append(('NB', GaussianNB()))
models_income.append(('SVM', SVC(gamma='auto')))

# Оцінка моделей
results_income = []
names_income = []

for name, model in models_income:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results_income.append(cv_results)
    names_income.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Візуалізація результатів
plt.boxplot(results_income, tick_labels=names_income)
plt.title('Algorithm Comparison for Income Data')
plt.savefig("income_algorithm_comparison.png")
