import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
import seaborn as sns

# Завантаження даних Iris
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=0)

# Створення та навчання Ridge класифікатора
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Оцінка класифікатора
print('Accuracy:', np.round(accuracy_score(y_test, y_pred), 4))
print('Precision:', np.round(precision_score(y_test, y_pred, average='weighted'), 4))
print('Recall:', np.round(recall_score(y_test, y_pred, average='weighted'), 4))
print('F1 Score:', np.round(f1_score(y_test, y_pred, average='weighted'), 4))
print('Cohen Kappa Score:', np.round(cohen_kappa_score(y_test, y_pred), 4))
print('Matthews Corrcoef:', np.round(matthews_corrcoef(y_test, y_pred), 4))
print('\t\tClassification Report:\n', classification_report(y_test, y_pred))

# Матриця плутанини
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig("Confusion.jpg")
plt.show()
