import pandas as pd

# Набір даних
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Перетворення набору даних в DataFrame
df = pd.DataFrame(data)

# Частотна таблиця
frequency_table = df.groupby(['Outlook', 'Humidity', 'Wind', 'Play']).size().unstack(fill_value=0)
print("Частотна таблиця:\n", frequency_table)

# Функція для створення таблиці правдоподібності
def calculate_likelihood(df, attribute, target):
    likelihood_table = pd.DataFrame(index=df[attribute].unique(), columns=df[target].unique())
    for attr_value in df[attribute].unique():
        for target_value in df[target].unique():
            likelihood_table.loc[attr_value, target_value] = len(df[(df[attribute] == attr_value) & (df[target] == target_value)]) / len(df[df[target] == target_value])
    return likelihood_table

# Таблиці правдоподібності для кожного атрибута
outlook_likelihood = calculate_likelihood(df, 'Outlook', 'Play')
humidity_likelihood = calculate_likelihood(df, 'Humidity', 'Play')
wind_likelihood = calculate_likelihood(df, 'Wind', 'Play')

print("Outlook Likelihood:\n", outlook_likelihood)
print("Humidity Likelihood:\n", humidity_likelihood)
print("Wind Likelihood:\n", wind_likelihood)
