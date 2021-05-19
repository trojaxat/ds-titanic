import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
titanic_data = os.path.join(CURR_DIR, "train.csv")
data_frame = pd.read_csv(titanic_data, index_col=0)
sns.set_style('darkgrid')

# 1. Calculate the number of surviving/non-surviving passengers and display it as a bar plot
# data['Survived'].value_counts().plot.bar()

# 2. Plot dead against survive rich
# first_class = data_frame.loc[data_frame['Pclass'] == 1]
# not_survived = first.loc[(first['survived']==1)]
# plot = first_class['Survived'].value_counts().plot.bar()
# plot.set_xlabel("Pclass")
# plot.set_ylabel("Survived")

# 3. Calculate the proportion of surviving 1st class passengers with regards to the total number of 1st class passengers
# (******************)

# 4. Create a bar plot with separate bars for male/female passengers and 1st/2nd/3rd class passengers
# sns.displot(data_frame, x="Pclass", hue="Sex", multiple="dodge", discrete=True)
# sns.countplot(x='sex', hue='pclass', data=df)

# 5. Create a histogram showing the age distribution of passengers. Compare surviving/non-surviving passengers
# sns.histplot(x="Age", hue="Survived", multiple="dodge", data=data_frame)

# 6. Calculate the average age for survived and drowned passengers separately
# plot = data_frame.groupby(["Survived"]).mean()['Age'].plot.bar()                                       

# 7. Replace missing age values by the mean age
mean = data_frame.groupby(["Survived"]).mean()['Age']
survived_mean = mean[0]
died_mean = mean[1]

# data_frame['Age'].fillna( died_mean, inplace=True)
# -> this doesnt work - data_frame['Age'] = data_frame.where(data_frame["Survived"].eq(0), NaN, died_mean)

# 8. Create a table counting the number of surviving/dead passengers separately for 1st/2nd/3rd class and male/female
# sns.histplot(x="Age", hue="Survived", multiple="dodge", data=data_frame)
new = data_frame.groupby(['Pclass', 'Survived', 'Sex']).size()
# sns.histplot(x="Survived", hue="Pclass", multiple="dodge", data=new)
pd.crosstab(data_frame['Survived'], [data_frame['Sex'], data_frame['Pclass']])


plt.show()
plt.close()