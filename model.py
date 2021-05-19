import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# Import data
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
titanic_data = os.path.join(CURR_DIR, "train.csv")
data_frame = pd.read_csv(titanic_data, index_col=0)

# # Imputer
# impute_and_scale = make_pipeline(SimpleImputer(), MinMaxScaler())
# impute_and_scale.fit(data_frame[['Age']])
# impute_and_scale.transform(data_frame[['Age']])

# Define Variable of interest
X = data_frame[[
    'Age', 
    'Fare',
    'Sex',
    'Parch', 
    'SibSp',
    'Embarked',
    'Pclass'
    ]]
y = data_frame['Survived']

numerical_x = [
    'Fare',
    'SibSp',
    'Parch'
]

catagorical_x = [
    'Sex',
    'Pclass',
    'Embarked'
]

bins_x = [
    'Age'
]

pipeline1 = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
    ])
    
pipeline2 = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(sparse=False))
    ])


pipeline3 = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('bins', KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='quantile')),
    ])
    
transformer = ColumnTransformer(
    transformers=[
        ('numerical', pipeline1, numerical_x),
        ('catagorical', pipeline2, catagorical_x),
        ('bins', pipeline3, bins_x),
        ('does_this_need_a_name', 'passthrough', ['Parch', 'SibSp'])
    ])


transformer.fit(X)
transformed_X = transformer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.25)

# Use dummy to test data, “stratified” generates predictions by respecting the training set’s class distribution.
model = LogisticRegression()
model.fit(transformed_X, y)
ypred = model.predict(X_train)
# Use model for prediction and scoring
score = model.score(X_test, y_test)
print('score: ', score)
ytrue = model.predict(y_test)

# ask about this
print(ytrue.shape)
# print('ytrue: ', model.summary() )
quit()
acc = accuracy_score(ytrue, ypred)
print('acc: ', acc)
recall = recall_score(ytrue, ypred, average='macro')
print('recall: ', recall)
f1 = f1_score(ytrue, ypred, average='macro')
print('f1: ', f1)
prec = precision_score(ytrue, ypred, average='macro')
print('prec: ', prec)
matrix = confusion_matrix(ytrue, ypred, labels=["ant", "bird", "cat"])
print('matrix: ', matrix)