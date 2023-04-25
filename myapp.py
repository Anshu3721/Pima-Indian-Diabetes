import numpy as np
import pickle
import streamlit as st

# Loading the saved model
# loaded_model = pickle.load(open('C:/Users/ANSHU/group_project/Pima indian diabetes streamlit/Diabetes_trained_model.sav', 'rb'))
loaded_model = pickle.load(open('Diabetes_trained_model.sav', 'rb'))
# Creating a function for prediction

def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():

    # giving a title
    st.title("Diabetes Prediction Web App")

    # Getting input data
    Pregnancies = st.text_input("No. of Pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood Pressure")
    Skinthickness = st.text_input("Skin Thickness value")
    BMI = st.text_input("Body Mass Index")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    age = st.text_input("Age")

    # Code for prediction
    diagnosis = ''

    # Creating a button for prediction
    if st.button("Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, Skinthickness, BMI, DiabetesPedigreeFunction, age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()