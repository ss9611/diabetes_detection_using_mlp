import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('mlp.keras')

# Define the feature names for input (you can adjust these based on your model's input structure)
feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']

# Function to preprocess user input and make a prediction
def predict_diabetes(input_data):
    # Convert input data into a numpy array and reshape it for prediction
    input_array = np.array([input_data])
    # Use the model to predict the outcome (0 = no diabetes, 1 = diabetes)
    prediction = model.predict(input_array)
    # Return the predicted class (rounding the probability to nearest integer)
    return np.round(prediction[0][0])

# Streamlit app design
st.title('Diabetes Detection App')
st.write('Provide patient data to predict whether they have diabetes.')

# Create input fields for each feature
input_data = []
for feature in feature_names:
    value = st.number_input(f'{feature}', min_value=0.0, format="%.2f")
    input_data.append(value)

# When the user clicks the "Predict" button
if st.button('Predict'):
    # Ensure all fields have been filled
    if any(val == 0.0 for val in input_data):
        st.error('Please fill in all fields.')
    else:
        # Predict the outcome based on user input
        prediction = predict_diabetes(input_data)
        if prediction == 1:
            st.success('The model predicts that the patient **has diabetes**.')
        else:
            st.success('The model predicts that the patient **does not have diabetes**.')

