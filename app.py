from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

 
model = pickle.load(open("salary_model.pkl", "rb"))
columns = pickle.load(open("salary_columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    
    input_data = {
        'Age': request.form.get('Age'),
        'Years of Experience': request.form.get('Years_of_Experience'),
        'Gender': request.form.get('Gender'),
        'Education Level': request.form.get('Education_Level'),
        'Job Title': request.form.get('Job_Title')
    }

   
    input_data['Age'] = int(input_data['Age']) if input_data['Age'] else 0
    input_data['Years of Experience'] = int(input_data['Years of Experience']) if input_data['Years of Experience'] else 0
    input_data['Gender'] = input_data['Gender'] if input_data['Gender'] else 'Male'
    input_data['Education Level'] = input_data['Education Level'] if input_data['Education Level'] else 'Bachelor'
    input_data['Job Title'] = input_data['Job Title'] if input_data['Job Title'] else 'Analyst'

   
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

     
    prediction = model.predict(df)
    return f"Predicted Salary: ${int(prediction[0])}"

if __name__ == "__main__":
    app.run(debug=True)