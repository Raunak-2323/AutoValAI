import streamlit as st
import pandas as pd
import pickle as pk

# Title
st.title("🚗 Car Price Prediction")

# Load model
try:
    with open('model.pkl', 'rb') as f:
        model = pk.load(f)
except:
    st.error("Model not found. Train the model first.")
    st.stop()

# Load brands and other categories
brands = ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
          'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
          'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar',
          'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force', 'Ambassador',
          'Ashok', 'Isuzu', 'Opel']

brand_mapping = {name: i+1 for i, name in enumerate(brands)}

# Input UI
name = st.selectbox("Car Brand", brands)
year = st.slider("Year of Manufacture", 1994, 2024, 2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
fuel = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'LPG', 'CNG'])
seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner',
                                    'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.slider("Mileage (km/l)", 10.0, 40.0, 18.0)
engine = st.slider("Engine (CC)", 700.0, 5000.0, 1500.0)
max_power = st.slider("Max Power (bhp)", 20.0, 200.0, 85.0)
seats = st.slider("Number of Seats", 2, 10, 5)

# Predict
if st.button("Predict Price"):
    input_df = pd.DataFrame([[brand_mapping[name], year, km_driven,
                              {'Diesel':1, 'Petrol':2, 'LPG':3, 'CNG':4}[fuel],
                              {'Individual':1, 'Dealer':2, 'Trustmark Dealer':3}[seller_type],
                              {'Manual':1, 'Automatic':2}[transmission],
                              {'First Owner':1, 'Second Owner':2, 'Third Owner':3,
                               'Fourth & Above Owner':4, 'Test Drive Car':5}[owner],
                              mileage, engine, max_power, seats]],
                            columns=['name', 'year', 'km_driven', 'fuel',
                                     'seller_type', 'transmission', 'owner',
                                     'mileage', 'engine', 'max_power', 'seats'])

    prediction = model.predict(input_df)
    st.success(f"💰 Estimated Price: ₹ {prediction[0]:,.2f}")
