# Importing essential libraries and modules
from flask import Flask, render_template, request, redirect, jsonify
from markupsafe import Markup
import numpy as np
import pandas as pd
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9
import config


# Loading nested disease translations
import json

with open('models/nested_disease.json', 'r', encoding='utf-8') as f:
    disease_translations = json.load(f)
# Importing translation dictionary
from translations import translations

# ==============================================================================================

# ------------------------- LOADING THE TRAINED MODELS ------------------------------------------

# Loading plant disease classification model
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# ==============================================================================================

# Weather API Integration
def weather_fetch(city_name):
    """
    Fetch and return the temperature and humidity of a city using OpenWeatherMap API
    :params: city_name
    :return: temperature (°C), humidity (%)
    """
    api_key = config.weather_api_key
    base_url = "https://api.openweathermap.org/data/2.5/weather?units=metric&q="

    complete_url = f"{base_url}{city_name}&appid={api_key}"
    
    try:
        response = requests.get(complete_url)
        data = response.json()

        if response.status_code == 200 and "main" in data:
            temperature = round(data["main"]["temp"], 2)
            humidity = data["main"]["humidity"]
            return temperature, humidity
        else:
            print(f"[ERROR] Weather fetch failed: {data.get('message')}")
            return None
    except Exception as e:
        print(f"[EXCEPTION] Error fetching weather: {e}")
        return None

# ==============================================================================================

# Image prediction function
def predict_image(img, model=disease_model):
    transform = transforms.Compose([ 
        transforms.Resize(256), 
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

# ==============================================================================================

# Flask App Initialization
app = Flask(__name__)

# Language selection (defaults to English)
@app.before_request
def before_request():
    # Get the language from the request, default to 'en'
    lang = request.cookies.get('lang', 'en')
    app.jinja_env.globals['lang'] = lang
    app.jinja_env.globals['t'] = translations[lang]  # Set 't' to the selected language's translations

# Home Page
@app.route('/')
def home():
    return render_template('index.html', title='Cultivora - Home')

# Crop Recommendation Page
@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html', title='Cultivora - Crop Recommendation')

# Fertilizer Recommendation Page
@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html', title='Cultivora - Fertilizer Suggestion')

# ==============================================================================================

# Crop Prediction Route
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        if weather_fetch(city):
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_recommendation_model.predict(data)[0]
            
            # Use translations for the crop name
            crop_name_translated = app.jinja_env.globals['t']['crop_name'].get(prediction, prediction)

            return render_template('crop-result.html', 
                                   prediction=crop_name_translated,  # Pass the translated crop name
                                   title='Cultivora - Crop Recommendation')
        else:
            return render_template('try_again.html', title='Cultivora - Crop Recommendation')

# Fertilizer Prediction Route
# Fertilizer Prediction Route
@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('Data/fertilizer.csv')
    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K

    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]

    if max_value == "N":
        key = 'NHigh' if n < 0 else 'Nlow'
    elif max_value == "P":
        key = 'PHigh' if p < 0 else 'Plow'
    else:
        key = 'KHigh' if k < 0 else 'Klow'

    # Use translations for fertilizer recommendation in the selected language
    language = request.cookies.get('lang', 'en')  # Default to 'en'
    recommendation = fertilizer_dic[key].get(language, fertilizer_dic[key]['en'])

    response = Markup(str(recommendation))
    return render_template('fertilizer-result.html', recommendation=response, title='Cultivora - Fertilizer Suggestion')


# Disease Prediction Route
# Disease Prediction Route
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title='Cultivora - Disease Detection')
        try:
            img = file.read()
            prediction = predict_image(img)

            # Fetch user's selected language
            language = request.cookies.get('lang', 'en')  # 'en' by default

            # Get translated disease name
            disease_name_translated = disease_translations.get(prediction, {}).get(language, prediction)

            prediction = Markup(str(disease_name_translated))
            return render_template('disease-result.html', prediction=prediction, title='Cultivora - Disease Detection')
        except Exception as e:
            print(f"Error predicting disease: {e}")
            pass
    return render_template('disease.html', title='Cultivora - Disease Detection')


# Language Switch Route
@app.route('/switch_language/<language>', methods=['GET'])
def switch_language(language):
    if language in translations:
        resp = redirect(request.referrer)
        resp.set_cookie('lang', language)
        return resp
    return redirect(request.referrer)

# ==============================================================================================

import os



if __name__ == "__main__":
    app.run(debug=True)