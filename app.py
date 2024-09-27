import streamlit as st
import numpy as np
import joblib 

# Load the scaler and model
scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")

# App Title
st.title("Real Estate Price Prediction App")

st.divider()

# User inputs for bedrooms, bathrooms, and size
bed = st.number_input("Enter the number of bedrooms", value=2, step=1, min_value=0)
bath = st.number_input("Enter the number of bathrooms", value=1, step=1, min_value=0)
size = st.number_input("Enter the size (sqft)", value=1000, step=50, min_value=1)

# Prepare the input for the model
X = [bed, bath, size]

st.divider()

# Button for triggering prediction
if st.button("Predict Price"):
    # Check inputs and run prediction
    if bed >= 0 and bath >= 0 and size > 0:
        st.balloons()
        
        # Scale the input and predict
        X_array = scaler.transform([X])
        prediction = model.predict(X_array)[0]
        
        # Display the predicted price
        st.write(f"The predicted price is ${prediction:,.2f}")
    else:
        st.write("Please ensure all inputs are valid.")
else:
    st.write("Please use the button for prediction.")