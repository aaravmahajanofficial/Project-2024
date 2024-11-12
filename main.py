from flask import Flask, request, jsonify
import numpy as np
import pickle
import json

app = Flask(__name__)

def churn_prediction(tenure, citytier, warehousetohome, gender, hourspendonapp, numberofdeviceregistered, satisfactionscore, maritalstatus, numberofaddress, complain, orderamounthikefromlastyear, couponused, ordercount, daysincelastorder, cashbackamount):
    # Load the model
    with open('churn_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the data columns from a JSON file
    with open("columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']

    # Prepare the input data
    input_data = [
        tenure, citytier, warehousetohome, gender,
        hourspendonapp, numberofdeviceregistered, satisfactionscore, maritalstatus,
        numberofaddress, complain, orderamounthikefromlastyear, couponused, ordercount, daysincelastorder, cashbackamount
    ]

    # Convert input_data to a dictionary with appropriate keys
    input_dict = {
        "tenure": tenure,
        "citytier": citytier,
        "warehousetohome": warehousetohome,
        "gender": gender,
        "hourspendonapp": hourspendonapp,
        "numberofdeviceregistered": numberofdeviceregistered,
        "satisfactionscore": satisfactionscore,
        "maritalstatus": maritalstatus,
        "numberofaddress": numberofaddress,
        "complain": complain,
        "orderamounthikefromlastyear": orderamounthikefromlastyear,
        "couponused": couponused,
        "ordercount": ordercount,
        "daysincelastorder": daysincelastorder,
        "cashbackamount": cashbackamount
    }

    # One-hot encode categorical variables
    for col in data_columns:
        if col in input_dict and isinstance(input_dict[col], str):
            input_dict[col] = input_dict[col].lower().replace(' ', '_')

    # Create a list of zeros for all columns
    input_array = np.zeros(len(data_columns))

    # Fill the input array with the values from input_dict
    for i, col in enumerate(data_columns):
        if col in input_dict:
            input_array[i] = input_dict[col]
        elif col in input_dict.keys():
            # One-hot encode the categorical variables
            if f"{col}_{input_dict[col]}" in data_columns:
                input_array[data_columns.index(f"{col}_{input_dict[col]}")] = 1

    # Predict the probability of churn
    output_probab = model.predict_proba([input_array])[0][1]
    return float(round(output_probab, 4))  # Round to 4 decimal places

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Extract the necessary values from the JSON data
        form_data = [
            data['Tenure'],
            data['Citytier'],
            data['Warehousetohome'],
            data['Gender'],
            data['Hourspendonapp'],
            data['Numberofdeviceregistered'],
            data['Satisfactionscore'],
            data['Maritalstatus'],
            data['Numberofaddress'],
            data['Complain'],
            data['Orderamounthikefromlastyear'],
            data['Couponused'],
            data['Ordercount'],
            data['Daysincelastorder'],
            data['Cashbackamount']
        ]

        # Get prediction
        output_probab = churn_prediction(*form_data)

        # Determine the churn status
        pred = "Churn" if output_probab > 0.4 else "Not Churn"

        # Return the result as JSON
        result = {
            'prediction': pred,
            'predict_probability': float(output_probab)
        }
        return jsonify(result), 200

    except Exception as e:
        # If an error occurs, return an error message
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
