import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Import data
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
titanic_data = os.path.join(CURR_DIR, "train.csv")
df = pd.read_csv(titanic_data, index_col=0)
df.dropna(inplace=True)
df.head()
print('shape', df.shape)

gender = {'male': 1,'female': 0}
survived = {0: 'Died', 1: 'Survived'}
df['Sex'] = [gender[item] for item in df['Sex']]
df['Survived'] = [survived[item] for item in df['Survived']]

y = df['Survived']
X = df[['Sex', 'Fare']]

# sns.scatterplot(x='Sex', y='Fare', hue='Survived', data=df)

predictions = []

for i, row in X.iterrows():
    if row['Sex'] == 'female':
        predictions.append('Survived')
    elif row['Fare'] < 16.8:
        predictions.append('Survived')
    else:
        predictions.append('Died')

acc_pred = accuracy_score(y, predictions)
print('acc: ', acc_pred)

m = DecisionTreeClassifier(max_depth=10)  # we allow that many questions
X_train, X_test, y_train, y_test = train_test_split(X, y)
m.fit(X_train, y_train)

ypred = m.predict(X_test)

acc_model = accuracy_score(ypred, y_test)
print('acc_model: ', acc_model)

# plt.figure(figsize=(12, 8))
print('m.classes_: ', m.classes_)
t = plot_tree(m, feature_names=['Sex', 'Fare'], filled=True, class_names=m.classes_)

plt.show()