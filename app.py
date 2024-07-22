import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

model = joblib.load('car_price_model(2).pkl')

feature_names = model.get_booster().feature_names

st.title('Car Price Evaluation')

year_made = st.number_input('Year Made', min_value=1980, max_value=2024, value=2010)
year_imported = st.number_input('Imported Year', min_value=1980, max_value=2024, value=2015)
odometer = st.number_input('Odometer (km)', min_value=0, value=50000)

brand = st.text_input('Brand')
model_car = st.text_input('Model')
color = st.selectbox('Color', ['Хар', 'Саарал', 'Бусад', 'Цагаан', 'Хар Саарал', 'Сувдан Цагаан'])
interior_color = st.selectbox('Interior Color', ['Саарал', 'Хар', 'Цагаан шар', 'Шаргал', 'Бор', 'Бусад'])
drive_type = st.selectbox('Drive Type', ['Бүх дугуй 4WD', 'Урдаа FWD', 'Хойноо RWD'])
transmission = st.selectbox('Transmission', ['Автомат', 'Механик'])
steering = st.selectbox('Steering', ['Буруу', 'Зөв'])
engine_type = st.selectbox('Engine Type', ['Бензин', 'Дизель', 'Хайбрид', 'Цахилгаан'])
leasing = st.selectbox('Leasing', ['Лизингтэй', 'Лизинггүй'])

input_data = {name: 0 for name in feature_names}

input_data['Year Made'] = year_made
input_data['Imported Year'] = year_imported
input_data['Odometer'] = odometer

input_data[f'Brand_{brand}'] = 1
input_data[f'Model_{model_car}'] = 1
input_data[f'Өнгө_{color}'] = 1
input_data[f'Дотор өнгө_{interior_color}'] = 1
input_data[f'Хөтлөгч_{drive_type}'] = 1
input_data[f'Хурдны хайрцаг_{transmission}'] = 1
input_data[f'Хүрд_{steering}'] = 1
input_data[f'Хөдөлгүүр_{engine_type}'] = 1
input_data[f'Лизинг_{leasing}'] = 1

input_df = pd.DataFrame([input_data])

for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
        
input_df = input_df[feature_names]

if st.button('Evaluate Price'):
    prediction = model.predict(input_df)
    st.write(f'Estimated Price: {prediction[0]:,.2f} сая ₮')
