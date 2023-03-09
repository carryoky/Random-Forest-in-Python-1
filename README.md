# Random-Forest-in-Python
There has never been a better time to get into machine learning. With the learning resources available online, free open-source tools with implementations of any algorithm imaginable, and the cheap availability of computing power through cloud services such as AWS, machine learning is truly a field that has been democratized by the internet. Anyone with access to a laptop and a willingness to learn can try out state-of-the-art algorithms in minutes. With a little more time, you can develop practical models to help in your daily life or at work (or even switch into the machine learning field and reap the economic benefits). This post will walk you through an end-to-end implementation of the powerful random forest machine learning model. It is meant to serve as a complement to my conceptual explanation of the random forest, but can be read entirely on its own as long as you have the basic idea of a decision tree and a random forest. A follow-up post details how we can improve upon the model built here.
Before we jump right into programming, we should lay out a brief guide to keep us on track. The following steps form the basis for any machine learning workflow once we have a problem and model in mind:

* 1.State the question and determine required data
* 2.Acquire the data in an accessible format
* 3.Identify and correct missing data points/anomalies as required
* 4.Prepare the data for the machine learning model
* 5.Establish a baseline model that you aim to exceed
* 6.Train the model on the training data
* 7.Make predictions on the test data
* 8.Compare predictions to the known test set targets and calculate performance metrics
* 9.If performance is not satisfactory, adjust the model, acquire more data, or try a different modeling technique
* 10.Interpret model and report results visually and numerically
## Steps
### Step 1: Importing necessary libraries and loading the dataset
```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the insurance dataset
df = pd.read_csv('/home/sam/Downloads/Churn_Modelling.csv')

# Preview the first 5 rows of the dataset
df.head()
```
### Step 2: Exploratory Data Analysis (EDA)
```python
# Check the data types of each column
df.dtypes

# Check the summary statistics of the numerical variables
df.describe()

# Check for missing values
df.isnull().sum()

# Plot a histogram of the age distribution
sns.histplot(data=df, x='Age', hue='Exited', kde=True, multiple='stack')
plt.title('Distribution of Age')
plt.show()

# Plot a box plots for the variable 'EstimatedSalary' based on the different categories of the selected columns in the list 'cols'
cols = ['Gender', 'Tenure', 'Age', 'Geography', 'HasCrCard']
for col in cols:
    plt.figure(figsize=(8,8))
    sns.boxplot(x = df[col], y = df['EstimatedSalary'])
    
    df.Geography = [1 if x == 'yes' else 0 for x in df.Geography]
df.Gender = [1 if x == 'male' else 0 for x in df.Gender]
df.corr()

sns.heatmap(df.corr())

# Plot a pairplot to visualize pairwise relationships between the features
sns.pairplot(df)


```
### Step 3: Training the Models
```python
# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

# Select relevant features for prediction
features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Split the data into features and target variable
X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
y = df['Exited']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error on the training set
y_train_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
print("Random Forest Mean Squared Error: {:.2f}".format(mse))

```
### Step 4: Combining the Models
```python
# Load the test data
test_data = pd.read_csv('churn_modelling.csv')

print(df.columns)

#  Chossing features for predicting the target variable
x = df

# Data split on df
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state=42)

# Create RandomForestRegressor model with max_features parameter
model = RandomForestRegressor(n_estimators=100, max_features=10, random_state=0)

# Train the model on the training data
rf.fit(X_train, y_train)

# Create a Random Forest regression model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Make predictions
y_pred = rf_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Random Forest Mean Squared Error:", mse)
```
### Step 5: Evaluating the Ensemble
```python# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the models
rf1 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
rf2 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
lr = LinearRegression()

# Define the ensemble models
voting_regressor = VotingRegressor([('rf1', rf1), ('rf2', rf2), ('lr', lr)])
stacking_regressor = StackingRegressor([('rf1', rf1), ('rf2', rf2)], final_estimator=lr)

# Fit the models
rf1.fit(X_train, y_train)
rf2.fit(X_train, y_train)
lr.fit(X_train, y_train)
voting_regressor.fit(X_train, y_train)
stacking_regressor.fit(X_train, y_train)

# Predict the test data using the models
y_pred_rf1 = rf1.predict(X_test)
y_pred_rf2 = rf2.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_voting = voting_regressor.predict(X_test)
y_pred_stacking = stacking_regressor.predict(X_test)

# Calculate the Mean Squared Error (MSE)
mse_rf1 = mean_squared_error(y_test, y_pred_rf1)
mse_rf2 = mean_squared_error(y_test, y_pred_rf2)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_voting = mean_squared_error(y_test, y_pred_voting)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)

# Print the MSE for each model
print("Random Forest 1 Mean Squared Error: {:.2f}".format(mse_rf1))
print("Random Forest 2 Mean Squared Error: {:.2f}".format(mse_rf2))
print("Linear Regression Mean Squared Error: {:.2f}".format(mse_lr))
print("Voting Regressor Mean Squared Error: {:.2f}".format(mse_voting))
print("Stacking Regressor Mean Squared Error: {:.2f}".format(mse_stacking))
```
### Step 6: Tuning the Ensemble
```python
# Importing the required libraries for evaluation
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the random forest model on the training data
model.fit(X_train, y_train)

# Predicting the target variable for the test data
y_pred_test = model.predict(X_test)

# Computing the mean squared error (MSE) on test data
mse = mean_squared_error(y_test, y_pred_test)
print("Random Forest Mean Squared Error:", mse)

# Computing the R-squared (R2) score on test data
r2 = r2_score(y_test, y_pred_test)
print("Random Forest R-squared Score:", r2)
```
