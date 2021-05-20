import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# This uses random forest to make many small decision trees and then aggregate the results

# Import data
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
titanic_data = os.path.join(CURR_DIR, "train.csv")
df = pd.read_csv(titanic_data, index_col=0)
df.dropna(inplace=True)
df.head()

# Change gender to binary for processing
gender = {'male': 1,'female': 0}
df['Sex'] = [gender[item] for item in df['Sex']]

# survived = {0: 'Died', 1: 'Survived'}
# df['Survived'] = [survived[item] for item in df['Survived']]

y = df['Survived']
X = df[['Sex', 'Fare']]

predictions = []

# My prediction
for i, row in X.iterrows():
    if row['Sex'] == 'female':
        predictions.append('Survived')
    elif row['Fare'] < 16.8:
        predictions.append('Survived')
    else:
        predictions.append('Died')

acc_pred = accuracy_score(y, predictions)
print('acc: ', acc_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y)
rfc = RandomForestClassifier(n_estimators=100, max_depth=2)    #Hyperparam: n_estimators(number of trees)
rfc.fit(X_train, y_train)

round = round(rfc.score(X_train, y_train),3)
print('round: ', round)

y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))

importance = rfc.feature_importances_
for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))

        fig = plt.figure(figsize=(8,6))

plt.bar([x for x in range(len(importance))], importance, alpha=0.5 )
plt.title('Random Forest Feature Importance')


plt.xticks(importance, ('Feature 0', 'Feature 1'))               
                                                                  
plt.ylabel('Score')
plt.show()

print(confusion_matrix( y_test, y_pred))
plot_confusion_matrix(rfc, X_test, y_test, cmap=plt.cm.Blues)

Xa=X.to_numpy()
ya=y.to_numpy()

fig = plt.figure(figsize=(8,6))

for clf in zip([rfc]):                
    rfc.fit(Xa, ya)
    fig = plot_decision_regions(X=Xa, y=ya, clf=rfc, legend=4)
plt.title('Decision Boundary')
plt.show()
