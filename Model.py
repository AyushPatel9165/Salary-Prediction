import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

 
data = pd.read_csv(r"C:\Data Science\Project Assignments and Project Topic list\Salary Prediction\Salary Data.csv")
 
numeric_cols = ['Age', 'Years of Experience', 'Salary']
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')  
        data[col] = data[col].fillna(data[col].mean())         

categorical_cols = ['Gender', 'Education Level', 'Job Title']
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].mode()[0])      

 
data = pd.get_dummies(data)
 
X = data.drop('Salary', axis=1)
y = data['Salary']

 
model = LinearRegression()
model.fit(X, y)

 
pickle.dump(model, open("salary_model.pkl", "wb"))
pickle.dump(X.columns, open("salary_columns.pkl", "wb"))

print("Model trained and saved successfully!")