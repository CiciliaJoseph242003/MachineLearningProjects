# Step 1: Upload and load the dataset
from google.colab import files
uploaded = files.upload()

import pandas as pd
import io

df = pd.read_csv(io.BytesIO(uploaded['Dataset .csv']))  # handles the space in filename
df.head()

df.columns

#To Drop unnecessary columns
drop_cols=['Restaurant ID', 'Restaurant Name', 'Address', 'Locality',
             'Locality Verbose', 'Switch to order menu', 'Rating color', 'Rating text']
df=df.drop(columns=drop_cols)

#Checking for missing values
print("Missing Values:\n",df.isnull().sum())

#Droping rows with any missing data for simplicity
df=df.dropna()

#Checking
print("After Dropping, missing values:\n",df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df.select_dtypes(include='object').columns:
  df[col]=le.fit_transform(df[col])

df.head()

#defininf X and y
X=df.drop('Aggregate rating', axis=1)
y=df['Aggregate rating']

#Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

#Linear Regression
lr=LinearRegression()
lr.fit(X_train, y_train)



#Decision Tree
dt=DecisionTreeRegressor()
dt.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

#Linear Regression
y_pred_lr=lr.predict(X_test)

#Decision Tree Preiction
y_pred_dt=dt.predict(X_test)

# Evaluation Function
def evaluate(y_test, y_pred, name):
    print(f"{name} Results:")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))
    print()

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_dt, "Decision Tree")

import seaborn as sns
import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Most Influential Features on Aggregate Rating")
plt.show()
