
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta

'''
This was a sideproject of mine in January 2025, which stem from a new years resolution that I want to be as happy as possible every single day in 2025. 
I naturally started with the hypothesis that I don't know what makes me happy on a given day and hence I need to run some tests.
I recorded data for 4 weeks and started with simple things: 
    1. Did I wake up early? (I always aim to wake up at 6:30, but it is very rare that I am able to fullfil it)
    2. Did I do my morning routine?
    3. Did I go to be early? (Aiming for 22:00 so I sleep by 22:30)
    4. Did I do Significant (1 hour) amount of house chores on the given day?
    5. Did I workout? (30 minutes)
    6. Did I read all my new personal emails?
    7. At the end of the day I also evaluated how happy was I in that day between 0 and 10

This is the datasource I am using currently, but after this little piece of code I am planning the next sprint to figure out what makes me happy on a given day.
'''

raw_data = pd.read_csv("How_was_my_day.csv")



#Have not added date to the original dataset so I am creating it here, from the knowledge that I made my first entry on the 1st of January 2025
start_date = datetime(2025, 1, 1)
date_list = [start_date + timedelta(days=i) for i in range(len(raw_data))]
date_strings = [date.strftime("%Y-%m-%d") for date in date_list]
raw_data["date"] = pd.to_datetime(date_strings)

#Rename columns to english
raw_data.rename(columns={
                    'keles' :'Wake up early',
                    'fekves' : 'Early bedtime',
                    'hazimunka' : 'House chores',
                    'edzes' : 'Workout',
                    'fogmosas' : 'Morning routine',
                    'pmail' : 'Personal emails checked'
                }, inplace=True)


# General overview of the data to make sure I haven't done anything wrong while recording it.
print(raw_data.info())  
print(raw_data.head())  


#Looking at distribution average and basic EDA
print(f"My average day was: {np.mean(raw_data['val']):.2f} good")

raw_data["val"].hist(bins=20, color="blue")
plt.xlabel("How good was my day?")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()
#Most of my days are 8s?


# See if my general mood has a trend and if yes is it upwards or downwards?
plt.figure(figsize=(10, 5))
plt.plot(raw_data["date"], raw_data["val"], marker="o", linestyle="-", color="b", label="Value")
plt.ylim(0, 10)
plt.xlabel("Date")
plt.ylabel("How good was my day?")
plt.title("Checking for overall trend")
plt.grid(True)
plt.legend()
plt.show()
#I don't suspect any strong trends in the dataset based on this line chart


# A heatmap to see the deeper connection between the variables I chose this time to analyise
corr = raw_data[['Wake up early', 'Early bedtime', 'House chores', 'Workout','Morning routine', 'Personal emails checked']].corr()  
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

'''
Results of EDA:
House chores and workout have a good (not strong) correlation does this mean if I start doing anything I will do multiple productive things? Like gatcha games which make you to have your very first very cheap purchase because repeated purchase is very likely? Interesting gamifying experiment idea.
I should have force myself to not check personal emails for a day to see what is their result...
Waking up early is detrimental for all other activities?
'''



'''
First question I wanted to tackle am I happier during weekends just because of the extra freetime?
In order to answer that question I used A/B testing
'''

raw_data["weekend_indicator"] = raw_data["date"].apply(lambda x: "Weekend" if x.weekday() >= 5 else "Weekday")
#Separated lists for weekday and weekend values
weekday_values = raw_data[raw_data["weekend_indicator"] == "Weekday"]["val"]
weekend_values = raw_data[raw_data["weekend_indicator"] == "Weekend"]["val"]

# Check for normality as that informs the decision on the A/B test 
weekday_normal = stats.shapiro(weekday_values).pvalue > 0.05
weekend_normal = stats.shapiro(weekend_values).pvalue > 0.05

print(weekday_normal,weekend_normal)
'''
As none of them are normally distributed, which is not a suprise given the short period of time analyised, I go with the mannwhitneyu test, which checks medians instead of averages
'''


def A_B_testing(A_values,B_values,alpha = 0.05):
    stat, p_value = stats.mannwhitneyu(A_values, B_values, alternative="two-sided")
    print(f"Test Statistic: {stat:.4f}, P-Value: {p_value:.4f}")
    if p_value < alpha:
        print("Result: Significant Difference between Data Sets")
    else:
        print("Result: No Significant Difference between Data Sets")

weekday_A, weekday_B = train_test_split(weekday_values, test_size=0.5, random_state=126)

#A/A test
A_B_testing(A_values = weekday_A,B_values = weekday_B)

#A/B test
A_B_testing(A_values = weekday_values,B_values = weekend_values)


'''
A/B test results there is no significant difference in my mood between weekdays and weekends.
'''


'''
Wanted to have a reminder above my bed how important certain things are for my happyness during the day so I decided to get a feature importance plot.
In order to do that I trained a simple CatBoostRegressor
'''
# Define features and target variable
X = raw_data.drop(columns=['val','date'])
y = raw_data['val']

# Identify categorical features
cat_features = ['Wake up early', 'Early bedtime', 'House chores', 'Workout', 'Morning routine', 'Personal emails checked','weekend_indicator'] 

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=126)

#Used simple CatBoostRegressor with low depth and learning rate as I am trying to avoid overfitting on a tiny dataset
model = CatBoostRegressor(
    iterations=1000,
    early_stopping_rounds=50,
    depth=5,              # Depth of each tree
    learning_rate=0.05,   # Learning rate
    loss_function="RMSE",  # Loss function for regression
    eval_metric="R2",   # Evaluation metric
    cat_features=cat_features,  # Define categorical features
    verbose=500
)

#Fitting the model
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

#Make the model make predictions
y_pred = model.predict(X_test)

#Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R2 Score: {r2:.4f}")


'''
Model results: The 0.6 R2 score indicates that important information is not captured in this dataset at the moment, but that is the point of the whole project to improve this
'''

#feature importance plot
feature_importances = model.get_feature_importance()
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Contribution to my personal wellbeing in a day")
plt.show()

'''
The next stage of this project is currently in planning
'''
