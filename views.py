from django.shortcuts import render
import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import json
import joblib

# Load your trained model and column names
model = pickle.load(open('diabetes.pickle', 'rb'))
columns = json.load(open('column_names.json', 'r'))['data_columns']

# Load the saved scalers
robust_scaler = joblib.load('robust_scaler.pkl')
standard_scaler = joblib.load('standard_scaler.pkl')

# Classification logic for insulin, glucose, and BMI
def classify_insulin(insulin):
    return "Normal" if 16 <= insulin <= 166 else "Abnormal"

def classify_glucose(glucose):
    if glucose > 126:
        return "Secret"
    elif 99 < glucose <= 126:
        return "Overweight"
    elif 70 < glucose <= 99:
        return "Normal"
    else:
        return "Low"

def classify_bmi(bmi):
    if bmi > 39.9:
        return "Obesity 3"
    elif 34.9 < bmi <= 39.9:
        return "Obesity 2"
    elif 29.9 < bmi <= 34.9:
        return "Obesity 1"
    elif 25 <= bmi <= 29.9:
        return "Overweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    else:
        return "Underweight"

# View to handle form submission and prediction
def predict_diabetes(request):
    if request.method == "POST":
        # Extract user input from the form
        user_input = {
            'Pregnancies': int(request.POST['Pregnancies']),
            'Glucose': float(request.POST['Glucose']),
            'BloodPressure': float(request.POST['BloodPressure']),
            'SkinThickness': float(request.POST['SkinThickness']),
            'Insulin': float(request.POST['Insulin']),
            'BMI': float(request.POST['BMI']),
            'DiabetesPedigreeFunction': float(request.POST['DiabetesPedigreeFunction']),
            'Age': int(request.POST['Age'])
        }

        # Classify insulin, glucose, and BMI
        insulin_class = classify_insulin(user_input['Insulin'])
        glucose_class = classify_glucose(user_input['Glucose'])
        bmi_class = classify_bmi(user_input['BMI'])

        # Prepare categorical features for prediction
        categorical_features = {
            "NewBMI_Obesity 1": 1 if bmi_class == "Obesity 1" else 0,
            "newbmi_Obesity 2": 1 if bmi_class == "Obesity 2" else 0,
            "newbmi_Obesity 3": 1 if bmi_class == "Obesity 3" else 0,
            "newbmi_Overweight": 1 if bmi_class == "Overweight" else 0,
            "NewInsulinScore_Normal": 1 if insulin_class == "Normal" else 0,
            "NewBMI_Normal": 1 if bmi_class == "Normal" else 0,
            "NewGlucose_Normal": 1 if glucose_class == "Normal" else 0,
            "NewGlucose_Overweight": 1 if glucose_class == "Overweight" else 0,
            "NewGlucose_Secret": 1 if glucose_class == "Secret" else 0,
        }

        # Prepare numerical features for prediction
        numerical_features = [user_input['Pregnancies'], user_input['Glucose'], user_input['BloodPressure'],
                              user_input['SkinThickness'], user_input['Insulin'], user_input['BMI'],
                              user_input['DiabetesPedigreeFunction'], user_input['Age']]

        # Combine categorical and numerical features
        feature_vector = numerical_features + [categorical_features.get(col, 0) for col in columns[8:]]

        # Convert to DataFrame for scaling and prediction
        input_df = pd.DataFrame([feature_vector], columns=columns)

        # Apply the scalers to the input features
        input_df.iloc[:, :8] = robust_scaler.transform(input_df.iloc[:, :8])
        input_df = standard_scaler.transform(input_df)
        input_df = pd.DataFrame(input_df, columns=columns)

        # Make the prediction
        prediction = model.predict(input_df.values)[0]
        return render(request, 'predict_diabetes.html', {'prediction': prediction, 'user_input': user_input})
    
    return render(request, 'predict_diabetes.html')
