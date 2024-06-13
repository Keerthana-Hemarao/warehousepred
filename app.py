from flask import Flask, request, render_template
import pandas as pd
import joblib
import datetime

# Load the trained model and encoders
model = joblib.load('logistic_regression_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    input_data = {
        'Creation Date': request.form['Creation_Date'],
        'Order Type': request.form['Order_Type'],
        'Service Type': request.form['Service_Type'],
        'Volume (ft3)': float(request.form['Volume_ft3']),
        'Weight (Lbs)': float(request.form['Weight_Lbs']),
        'Sales Value ($)': float(request.form['Sales_Value']),
        'Nr Lines': float(request.form['Nr_Lines']),
        'Distance WH01': float(request.form['Distance_WH01']),
        'Distance WH02': float(request.form['Distance_WH02']),
        'Parameter 1 WHS Utilization': float(request.form['Parameter_1_WHS_Utilization']),
        'Parameter 1* WHS Utilization': float(request.form['Parameter_1_Star_WHS_Utilization']),
        'Parameter 2 Sales price': float(request.form['Parameter_2_Sales_price']),
        'Parameter 2* Sales price': float(request.form['Parameter_2_Star_Sales_price']),
        'Parameter 3 LeadTime': float(request.form['Parameter_3_LeadTime']),
        'Parameter 3* LeadTime': float(request.form['Parameter_3_Star_LeadTime']),
        'Parameter 4 Carrier Cost': float(request.form['Parameter_4_Carrier_Cost']),
        'Parameter 4* Carrier Cost': float(request.form['Parameter_4_Star_Carrier_Cost']),
    }

    input_df = pd.DataFrame([input_data])

    # Convert 'Creation Date' to datetime and then drop it
    input_df['Creation Date'] = pd.to_datetime(input_df['Creation Date'])
    input_df = input_df.drop(['Creation Date'], axis=1)

    # Encode categorical input features
    for column in input_df.select_dtypes(include=['object']).columns:
        input_df[column] = label_encoders[column].transform(input_df[column])

    # Predict the final result warehouse
    predicted_numeric = model.predict(input_df)
    predicted_label = target_encoder.inverse_transform(predicted_numeric)

    return render_template('index.html', prediction_text=f'Predicted Final Result warehouse: {predicted_label[0]}')

if __name__ == '__main__':
    app.run(debug=True)
