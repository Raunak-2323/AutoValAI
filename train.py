import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle as pk

# Load dataset
df = pd.read_csv('Cardetails.csv')
df.drop(columns=['torque'], inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Extract brand
df['name'] = df['name'].apply(lambda x: x.split()[0].strip())

# Clean numeric columns
def clean(value):
    try:
        return float(value.split()[0])
    except:
        return 0.0

df['mileage'] = df['mileage'].apply(clean)
df['engine'] = df['engine'].apply(clean)
df['max_power'] = df['max_power'].apply(clean)

# Encode categories
brand_mapping = {v: i+1 for i, v in enumerate(df['name'].unique())}
df['name'] = df['name'].map(brand_mapping)

df['fuel'].replace({'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4}, inplace=True)
df['seller_type'].replace({'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}, inplace=True)
df['transmission'].replace({'Manual': 1, 'Automatic': 2}, inplace=True)
df['owner'].replace({'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
                     'Fourth & Above Owner': 4, 'Test Drive Car': 5}, inplace=True)

# Split data
X = df.drop('selling_price', axis=1)
y = df['selling_price']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pk.dump(model, f)

print("✅ Model trained and saved as model.pkl")
