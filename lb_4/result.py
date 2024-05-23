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

# Ймовірність P(Play=Yes) та P(Play=No)
P_Play_Yes = len(df[df['Play'] == 'Yes']) / len(df)
P_Play_No = len(df[df['Play'] == 'No']) / len(df)

# Ймовірності для заданих умов (Outlook = Overcast, Humidity = High, Wind = Strong)
P_Outlook_Overcast_Yes = outlook_likelihood.loc['Overcast', 'Yes']
P_Humidity_High_Yes = humidity_likelihood.loc['High', 'Yes']
P_Wind_Strong_Yes = wind_likelihood.loc['Strong', 'Yes']

P_Outlook_Overcast_No = outlook_likelihood.loc['Overcast', 'No']
P_Humidity_High_No = humidity_likelihood.loc['High', 'No']
P_Wind_Strong_No = wind_likelihood.loc['Strong', 'No']

# Обчислення ймовірності для Play=Yes
P_Yes = P_Outlook_Overcast_Yes * P_Humidity_High_Yes * P_Wind_Strong_Yes * P_Play_Yes

# Обчислення ймовірності для Play=No
P_No = P_Outlook_Overcast_No * P_Humidity_High_No * P_Wind_Strong_No * P_Play_No

# Нормалізація
P_Play_Yes_Given_Conditions = P_Yes / (P_Yes + P_No)
P_Play_No_Given_Conditions = P_No / (P_Yes + P_No)

print(f"Ймовірність, що гра відбудеться: {P_Play_Yes_Given_Conditions}")
print(f"Ймовірність, що гра не відбудеться: {P_Play_No_Given_Conditions}")
