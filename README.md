Diabetes Prediction  

 Project Overview  
The Diabetes Prediction project is a Django-based web application that allows users to input their health parameters (such as insulin levels, glucose, BMI, etc.) and predicts the likelihood of developing diabetes. The backend uses machine learning models to make predictions based on the user's input. Data preprocessing is performed on the dataset to ensure accurate predictions, and the performance of the model is evaluated using metrics like accuracy and AUC.

 Features  
- Web interface for users to input health parameters.  
- Predicts whether an individual has diabetes based on input features using a trained machine learning model.  
- Implements multiple machine learning models for classification.  
- Evaluates model performance using accuracy and AUC.  
- Data preprocessing is handled with Pandas and Scikit-learn.  

 Technologies Used  
- Django: For web application development.  
- Machine Learning: Logistic Regression, Random Forest, SVM, etc.  
- Data Preprocessing: Pandas, NumPy, Scikit-learn.  
- Evaluation Metrics: Accuracy,  AUC  
- Model Serialization: Pickle (to save the trained model).  

 Data Handling and Preprocessing  
Data preprocessing was performed to ensure the quality of the dataset:  
- Handling Missing Values: Cleaned the dataset by addressing missing or incomplete data.  
- Feature Scaling: StandardScaler was applied to scale numerical features.  
- Data Encoding: Categorical variables were encoded appropriately.  
- Splitting Dataset: The data was split into training and testing sets to evaluate model performance.  

 Model Evaluation  
The performance of the machine learning models was evaluated using the following metrics:  
- Accuracy: The percentage of correct predictions.  
- F1 Score: The harmonic mean of precision and recall.  
- AUC (Area Under the Curve): Measures model performance across different thresholds.  

 Setup Instructions  
     
1. Create a virtual environment:  
   bash  
   python -m venv venv  
   source venv/bin/activate   On Windows: venv\Scripts\activate  
     

2. Install the required dependencies:  
   bash  
   pip install -r requirements.txt  
     

3. Run the Django migrations to set up the database:  
   bash  
   python manage.py migrate  
     

4. Start the Django development server:  
   bash  
   python manage.py runserver  
     

5. Access the web application at:  
   bash  
   http://127.0.0.1:8000  
     

 Project Structure  
  
diabetes-prediction/  
├── manage.py                     Django management script  
├── diabetes_app/                  Main app folder  
│   ├── migrations/                Database migrations  
│   ├── templates/                 HTML files  
│   │   ├── home.html              Main page for user input  
│   ├── static/                    Static files (CSS, JS)  
│   ├── models.py                  (Optional) Django models  
│   ├── views.py                   Handles logic for predictions  
│   ├── urls.py                    URL routing  
│   ├── forms.py                   Django form for user input  
├── diabetes_prediction/           Project folder  
│   ├── settings.py                Django settings  
│   ├── urls.py                    Root URL configuration  
│   ├── wsgi.py                    WSGI configuration  
├── model/                         Folder for saving the trained model  
│   ├── model.pkl                  Trained model file  
├── requirements.txt               List of required Python libraries  
└── README.md                      Project documentation  
  

 Usage  
1. Input health parameters like insulin, glucose, BMI, and others on the web form.  
2. The model will predict the likelihood of diabetes based on the user's input.  
3. View the prediction result and the model's performance metrics.  

