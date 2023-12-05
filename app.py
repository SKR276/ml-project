import pandas as pd
from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')


def preprocess_categorical(df):
    label_encoder = LabelEncoder()

    categorical_columns = ['Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                           'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'StreamingTV']

    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

    return df


def preprocess(df):
    # One-hot encode categorical columns
    df = preprocess_categorical(df)
    categorical_cols = ['Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'StreamingTV']
    df_categorical = pd.get_dummies(df[categorical_cols])

    # Drop original categorical columns and concatenate the one-hot encoded columns
    df = pd.concat([df.drop(categorical_cols, axis=1), df_categorical], axis=1)

    # Feature scaling for numerical columns
    scaler = MinMaxScaler()
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Fill missing values if any
    df.fillna(0, inplace=True)  # Replace NaNs with 0

    return df


def predict_churn_lr(input_data):
    print(f"Input Data Shape: {input_data.shape}")  # Check input data shape
    # Ensure input data is in numeric format
    relevant_features = ['SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'StreamingTV']

    # Ensure only relevant features are used for prediction
    input_data = input_data[relevant_features]

    print(f"Selected Features Shape: {input_data.shape}")  # Check selected features shape

    # Your prediction logic here
    predicted_result = model.predict(input_data)
    print(f"Predicted Result: {predicted_result}")  # Check predicted result
    return predicted_result


def collect_feedback(feedback_choice, detailed_feedback):
    print("We value your feedback! Please take a moment to share your thoughts.")
    print("1. How likely are you to churn?")
    print("2. What factors influenced your decision?")
    print("3. Any specific services or features you'd like to see improved?")
    print("4. General comments or suggestions")

    feedback_data = {
        'Feedback Choice': feedback_choice,
        'Feedback': detailed_feedback
    }

    print("Thank you for sharing your feedback!")
    return feedback_data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input_data = {
            'SeniorCitizen': [int(request.form['senior_citizen'])],
            'Dependents': [request.form['dependents']],
            'tenure': [int(request.form['tenure'])],
            'PhoneService': [request.form['phone_service']],
            'MultipleLines': [request.form['multiple_lines']],
            'InternetService': [request.form['internet_service']],
            'OnlineSecurity': [request.form['online_security']],
            'OnlineBackup': [request.form['online_backup']],
            'TechSupport': [request.form['tech_support']],
            'StreamingTV': [request.form['streaming_tv']],
            'MonthlyCharges': [float(request.form['monthly_charges'])],
            'TotalCharges': [float(request.form['total_charges'])]

        }

        # Create a DataFrame from user input data
        features_df = pd.DataFrame(user_input_data, index=[0])

        # Perform preprocessing on the input data
        preprocessed_input = preprocess(features_df)

        # Make the prediction using preprocessed data
        prediction = predict_churn_lr(preprocessed_input)

        return render_template('result.html', prediction=prediction)


@app.route('/give_feedback', methods=['GET', 'POST'])
def give_feedback():
    if request.method == 'POST':
        feedback_choice = request.form.get('feedback_choice')
        detailed_feedback = request.form.get('detailed_feedback')
        print(f"Feedback Choice: {feedback_choice}, Detailed Feedback: {detailed_feedback}")
        return "Thank you for your feedback!"
    return render_template('feedback.html')


if __name__ == '__main__':
    app.run(debug=True)
