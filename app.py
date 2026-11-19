import os
import json
import requests
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Smart Agriculture System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Base Paths & Model Caching
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
CROP_MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommendation_model.pkl")

DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "title": "Bacterial Spot",
        "severity": "Medium",
        "description": "Caused by Xanthomonas bacteria, producing small, dark water-soaked spots on leaves and fruit.",
        "treatment": "Apply copper-based fungicides/bactericides. Avoid overhead irrigation and work in fields only when foliage is dry."
    },
    "Tomato___Early_blight": {
        "title": "Early Blight",
        "severity": "Medium",
        "description": "Caused by Alternaria solani, characterized by concentric dark rings ('target spots') on older leaves.",
        "treatment": "Prune lower infected leaves, mulch around plant bases to prevent soil splash, and apply chlorothalonil or copper spray."
    },
    "Tomato___Late_blight": {
        "title": "Late Blight",
        "severity": "High (Urgent)",
        "description": "Caused by Phytophthora infestans. Leaves show water-soaked lesions that rapidly turn brown/black with white mold under humid conditions.",
        "treatment": "Remove and securely dispose of infected plants immediately. Apply preventive fungicides like mancozeb or copper hydroxide."
    },
    "Tomato___Leaf_Mold": {
        "title": "Leaf Mold",
        "severity": "Low-Medium",
        "description": "Caused by Passalora fulva. Causes pale greenish-yellow spots on upper leaf surfaces and olive-green velvety mold on undersides.",
        "treatment": "Improve air circulation, space plants well, avoid wet leaves, and lower greenhouse humidity below 85%."
    },
    "Tomato___Septoria_leaf_spot": {
        "title": "Septoria Leaf Spot",
        "severity": "Medium",
        "description": "Caused by Septoria lycopersici. Small, circular spots with dark brown margins and gray centers appear on lower leaves.",
        "treatment": "Remove lower infected foliage, avoid overhead watering, rotate crops every 2-3 years, and apply protective copper fungicide."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "title": "Two-Spotted Spider Mite Infestation",
        "severity": "Medium",
        "description": "Tiny arachnids causing yellow stippling/speckling on leaves along with delicate fine silk webbing underneath.",
        "treatment": "Spray insecticidal soap or horticultural neem oil. Increase ambient humidity and introduce beneficial predatory mites."
    },
    "Tomato___Target_Spot": {
        "title": "Target Spot",
        "severity": "Medium",
        "description": "Caused by Corynespora cassiicola, leading to pinpoint brown lesions that expand with concentric rings.",
        "treatment": "Promote airflow through pruning, avoid overhead watering, and apply labeled protectant fungicides."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "title": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "severity": "High",
        "description": "Transmitted by whiteflies (Bemisia tabaci). Symptoms include severe leaf curling upward, stunted plant growth, and yellow margins.",
        "treatment": "Manage whitefly vectors with yellow sticky traps, insect mesh netting, and insecticidal soaps. Rogue out infected plants."
    },
    "Tomato___Tomato_mosaic_virus": {
        "title": "Tomato Mosaic Virus (ToMV)",
        "severity": "High",
        "description": "Mechanically transmitted virus causing mottled light and dark green patterns, deformed leaves, and reduced yields.",
        "treatment": "No cure once infected; immediately remove and destroy infected plants. Sanitize pruning shears and wash hands before handling crops."
    },
    "Tomato___healthy": {
        "title": "Healthy Leaf",
        "severity": "None",
        "description": "No visible disease symptoms. The leaf shows healthy pigmentation, intact cellular structure, and robust foliage.",
        "treatment": "Maintain optimal irrigation, balanced N-P-K fertilization, and routine preventative scouting."
    }
}

@st.cache_resource(show_spinner="Loading AI Models...")
def load_models():
    """Load machine learning and deep learning models once into cache."""
    try:
        disease = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    except Exception as e:
        disease = None
        st.error(f"Error loading disease detection model: {e}")

    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            classes = json.load(f)
    except Exception as e:
        classes = []
        st.error(f"Error loading class names: {e}")

    try:
        crop = joblib.load(CROP_MODEL_PATH)
    except Exception as e:
        crop = None
        st.error(f"Error loading crop recommendation model: {e}")

    return disease, classes, crop

disease_model, class_names, crop_model = load_models()

# ---------------------------------------------------------
# Sidebar Navigation & App Info
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌾 Smart Agriculture AI")
    st.markdown("Precision AI tools for farmers and agronomists.")
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        [
            "Home",
            "Crop Recommendation",
            "Disease Detection",
            "Soil Monitoring",
            "Weather Monitoring",
            "Farming Insights"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📊 System Status")
    if crop_model is not None and disease_model is not None:
        st.success("🟢 All AI Models Online")
    else:
        st.warning("⚠️ Some Models Offline")
        
    st.caption("Powered by TensorFlow, Scikit-learn & Open-Meteo")

# ---------------------------------------------------------
# View 1: Home
# ---------------------------------------------------------
if menu == "Home":
    st.title("🌱 AI Smart Agriculture Platform")
    st.markdown(
        "Empowering farmers with state-of-the-art **machine learning crop recommendation**, "
        "**deep learning plant disease diagnosis**, and **real-time weather intelligence**."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🌾 Crop Recommendation")
        st.write("Predict the ideal crop to cultivate based on Nitrogen, Phosphorus, Potassium, soil pH, temperature, and rainfall.")
        st.info("Utilizes a high-accuracy Random Forest Classifier trained on agricultural datasets.")

    with col2:
        st.markdown("### 🍅 Disease Detection")
        st.write("Diagnose tomato crop diseases from leaf photographs with instant AI classification and expert remedial steps.")
        st.info("Powered by a deep Convolutional Neural Network (MobileNetV2) trained on PlantVillage.")

    with col3:
        st.markdown("### 🌦️ Weather & Soil Intel")
        st.write("Live local weather forecasts, 7-day precipitation outlooks, and soil parameter diagnostics.")
        st.info("Integrated with Open-Meteo real-time global meteorological telemetry.")

    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")
    st.markdown(
        """
        1. **Crop Recommendation**: Go to the sidebar, enter your soil nutrients and climate parameters, and click **Recommend Crop**.
        2. **Disease Detection**: Upload an image of a tomato leaf to detect common blights, viruses, or confirm healthy foliage.
        3. **Soil Monitoring**: Input your soil pH and moisture to evaluate soil health and recommended soil amendments.
        4. **Weather Monitoring**: Type your city or village name to get real-time temperatures, wind speed, and rain probability.
        """
    )

# ---------------------------------------------------------
# View 2: Crop Recommendation
# ---------------------------------------------------------
elif menu == "Crop Recommendation":
    st.title("🌾 AI Crop Recommendation")
    st.write("Enter soil nutrient concentrations and climate data to predict the best-suited crop for your land.")
    st.markdown("---")

    if crop_model is None:
        st.error("Crop recommendation model is not loaded. Please verify models/crop_recommendation_model.pkl.")
    else:
        col_preset, _ = st.columns([2, 3])
        with col_preset:
            preset = st.selectbox(
                "💡 Optional: Load sample soil profile",
                ["Manual Input", "Rice / Paddy Profile", "Cotton Profile", "Chickpea / Legume Profile", "Maize / Corn Profile"]
            )

        defaults = {
            "Manual Input": (90.0, 42.0, 43.0, 24.5, 75.0, 6.5, 180.0),
            "Rice / Paddy Profile": (85.0, 45.0, 40.0, 26.0, 82.0, 6.8, 240.0),
            "Cotton Profile": (120.0, 40.0, 20.0, 25.0, 60.0, 7.5, 75.0),
            "Chickpea / Legume Profile": (40.0, 65.0, 80.0, 19.0, 16.0, 7.2, 70.0),
            "Maize / Corn Profile": (75.0, 48.0, 20.0, 23.0, 65.0, 6.2, 95.0),
        }
        init_n, init_p, init_k, init_temp, init_hum, init_ph, init_rain = defaults[preset]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🧪 Soil Nutrient Levels (kg/ha)")
            n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=300.0, value=float(init_n), step=1.0)
            p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=300.0, value=float(init_p), step=1.0)
            k = st.number_input("Potassium (K)", min_value=0.0, max_value=300.0, value=float(init_k), step=1.0)
            ph = st.number_input("Soil pH", min_value=1.0, max_value=14.0, value=float(init_ph), step=0.1)

        with col2:
            st.markdown("#### 🌤️ Climate & Environmental Factors")
            temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=float(init_temp), step=0.5)
            humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=float(init_hum), step=1.0)
            rainfall = st.number_input("Annual / Seasonal Rainfall (mm)", min_value=0.0, max_value=3000.0, value=float(init_rain), step=5.0)

        st.markdown(" ")
        if st.button("🔮 Recommend Best Crop", type="primary"):
            values = pd.DataFrame(
                [[n, p, k, temp, humidity, ph, rainfall]],
                columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            )

            try:
                prediction = crop_model.predict(values)[0]
                crop_name = str(prediction).capitalize()

                st.markdown("---")
                st.success(f"### 🏆 Optimal Recommended Crop: **{crop_name}**")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Selected N-P-K Ratio", f"{int(n)} : {int(p)} : {int(k)}")
                res_col2.metric("Soil pH Classification", "Slightly Acidic" if ph < 6.5 else ("Neutral/Optimal" if ph <= 7.5 else "Alkaline"))
                res_col3.metric("Rainfall Classification", "High" if rainfall > 150 else ("Moderate" if rainfall > 75 else "Low / Arid"))

                st.info(f"🌾 Based on the environmental conditions and nutrient availability, cultivating **{crop_name}** yields the highest predicted viability and yield efficiency.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------
# View 3: Disease Detection
# ---------------------------------------------------------
elif menu == "Disease Detection":
    st.title("🍅 AI Tomato Plant Disease Detection")
    st.write("Upload a photograph of a tomato crop leaf to identify diseases and receive instant remediation guidance.")
    st.markdown("---")

    if disease_model is None or not class_names:
        st.error("Disease detection model or class list is unavailable. Please check the models directory.")
    else:
        uploaded = st.file_uploader(
            "Upload a clear image of a tomato leaf (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            col_img, col_pred = st.columns([1, 1])

            with col_img:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

            with col_pred:
                # Preprocess image
                resized = image.resize((224, 224))
                arr = np.array(resized, dtype=np.float32)
                arr = np.expand_dims(arr, axis=0)
                arr = preprocess_input(arr)

                if st.button("🔬 Analyze Leaf Health", type="primary"):
                    with st.spinner("Processing image through MobileNetV2 neural network..."):
                        predictions = disease_model.predict(arr)
                        index = int(np.argmax(predictions[0]))
                        confidence = float(predictions[0][index]) * 100
                        raw_label = class_names[index] if index < len(class_names) else "Unknown"

                        info = DISEASE_INFO.get(raw_label, {
                            "title": raw_label.replace("Tomato___", "").replace("_", " "),
                            "severity": "Unknown",
                            "description": "Diagnosis completed.",
                            "treatment": "Consult a certified local agricultural extension officer."
                        })

                        if "healthy" in raw_label.lower():
                            st.success(f"### Result: {info['title']}")
                        else:
                            st.error(f"### Detected: {info['title']}")

                        st.progress(min(1.0, confidence / 100.0))
                        st.write(f"**Diagnostic Confidence:** `{confidence:.2f}%`")
                        st.write(f"**Severity Level:** `{info['severity']}`")

                        st.markdown("#### 🩺 Symptoms & Description")
                        st.write(info["description"])

                        st.markdown("#### 💊 Recommended Agricultural Treatment")
                        st.info(info["treatment"])

                        st.warning("⚠️ **Note:** Machine learning classifications are advisory. Verify symptoms with on-ground agricultural specialists.")

# ---------------------------------------------------------
# View 4: Soil Monitoring
# ---------------------------------------------------------
elif menu == "Soil Monitoring":
    st.title("🪴 Soil Health & Moisture Monitoring")
    st.write("Evaluate soil conditions, nutrient balances, and get automated recommendations for soil amendments.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        soil_ph = st.slider("Soil pH Level", min_value=3.5, max_value=10.0, value=6.5, step=0.1)
        soil_moisture = st.slider("Current Soil Moisture Content (%)", min_value=5, max_value=95, value=40, step=1)
        organic_matter = st.slider("Organic Matter Content (%)", min_value=0.5, max_value=10.0, value=3.2, step=0.1)

    with col2:
        soil_texture = st.selectbox("Soil Texture Type", ["Loamy (Ideal)", "Sandy", "Clay", "Silt", "Peat"])
        n_status = st.select_slider("Nitrogen Level", options=["Deficient", "Adequate", "Excessive"], value="Adequate")
        p_status = st.select_slider("Phosphorus Level", options=["Deficient", "Adequate", "Excessive"], value="Adequate")
        k_status = st.select_slider("Potassium Level", options=["Deficient", "Adequate", "Excessive"], value="Adequate")

    st.markdown("---")
    st.subheader("📊 Soil Diagnostic Report")

    # pH classification
    if soil_ph < 5.5:
        ph_status = "Strongly Acidic 🔴"
        ph_action = "Apply agricultural lime (calcium carbonate) or dolomite to raise soil pH to 6.2 - 6.8."
    elif soil_ph < 6.3:
        ph_status = "Moderately Acidic 🟡"
        ph_action = "Add composted organic manure and light lime treatment."
    elif soil_ph <= 7.3:
        ph_status = "Optimal / Neutral 🟢"
        ph_action = "Soil pH is in the prime nutrient absorption range for the majority of food crops."
    elif soil_ph <= 8.2:
        ph_status = "Moderately Alkaline 🟡"
        ph_action = "Incorporate elemental sulfur or gypsum to reduce alkalinity and improve micronutrient availability."
    else:
        ph_status = "Highly Alkaline 🔴"
        ph_action = "Heavy application of soil acidifiers, peat moss, or sulfur required. Leach excess soluble salts."

    # Moisture classification
    if soil_moisture < 25:
        moist_status = "Dry / Drought Stress ⚠️"
        moist_action = "Immediate irrigation required. Consider drip irrigation and organic mulching to retain soil moisture."
    elif soil_moisture <= 65:
        moist_status = "Optimal Moisture 💧"
        moist_action = "Moisture level is well-balanced for root aeration and water uptake."
    else:
        moist_status = "Waterlogged / Saturated 🚨"
        moist_action = "Risk of root hypoxia and fungal rot. Improve field drainage ditches and suspend irrigation."

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("pH Assessment", ph_status)
    col_b.metric("Moisture Assessment", moist_status)
    col_c.metric("Organic Matter Status", "Good" if organic_matter >= 3.0 else "Needs Enrichment")

    st.markdown("#### 🚜 Agronomist Remediation Checklist")
    st.markdown(f"- **pH Remediation:** {ph_action}")
    st.markdown(f"- **Moisture Guidance:** {moist_action}")
    if organic_matter < 2.5:
        st.markdown("- **Organic Matter:** Low organic carbon detected. Apply well-rotted farmyard manure (FYM) or green manure cover crops.")
    if n_status == "Deficient":
        st.markdown("- **Nitrogen Boost:** Incorporate urea, ammonium sulfate, or legume cover crops.")
    if p_status == "Deficient":
        st.markdown("- **Phosphorus Boost:** Apply Single Super Phosphate (SSP) or bone meal near the root zone.")
    if k_status == "Deficient":
        st.markdown("- **Potassium Boost:** Apply Muriate of Potash (MOP) or wood ash.")

# ---------------------------------------------------------
# View 5: Weather Monitoring
# ---------------------------------------------------------
elif menu == "Weather Monitoring":
    st.title("🌦️ Live Meteorological Monitoring")
    st.write("Real-time weather observation and 7-day agricultural forecasts powered by Open-Meteo telemetry.")
    st.markdown("---")

    city = st.text_input("Enter City / Region Name", "Bengaluru")

    if st.button("Fetch Live Weather Intelligence", type="primary"):
        with st.spinner(f"Querying meteorological satellite telemetry for {city}..."):
            try:
                # 1. Geocoding
                geo_url = "https://geocoding-api.open-meteo.com/v1/search"
                geo_response = requests.get(
                    geo_url,
                    params={"name": city, "count": 1},
                    timeout=15
                )
                geo_response.raise_for_status()
                geo_data = geo_response.json()

                if not geo_data.get("results"):
                    st.error(f"Location '{city}' could not be located. Please check the spelling.")
                else:
                    place = geo_data["results"][0]
                    latitude = place["latitude"]
                    longitude = place["longitude"]

                    # 2. Weather Forecast
                    weather_url = "https://api.open-meteo.com/v1/forecast"
                    params = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "current": [
                            "temperature_2m",
                            "relative_humidity_2m",
                            "apparent_temperature",
                            "precipitation",
                            "wind_speed_10m",
                            "weather_code"
                        ],
                        "daily": [
                            "temperature_2m_max",
                            "temperature_2m_min",
                            "precipitation_probability_max"
                        ],
                        "timezone": "auto",
                        "forecast_days": 7
                    }

                    response = requests.get(weather_url, params=params, timeout=20)
                    response.raise_for_status()
                    data = response.json()

                    current = data["current"]

                    st.success(f"📍 Weather station: **{place['name']}, {place.get('country', '')}** (Lat: {latitude:.2f}, Lon: {longitude:.2f})")

                    # Current Metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Ambient Temperature", f"{current['temperature_2m']} °C")
                    c2.metric("Relative Humidity", f"{current['relative_humidity_2m']} %")
                    c3.metric("Feels Like", f"{current['apparent_temperature']} °C")

                    c4, c5 = st.columns(2)
                    c4.metric("Current Precipitation", f"{current['precipitation']} mm")
                    c5.metric("Wind Velocity (10m)", f"{current['wind_speed_10m']} km/h")

                    weather_codes = {
                        0: "Clear Sky ☀️",
                        1: "Mainly Clear 🌤️",
                        2: "Partly Cloudy ⛅",
                        3: "Overcast ☁️",
                        45: "Fog 🌫️",
                        48: "Depositing Rime Fog 🌫️",
                        51: "Light Drizzle 🌦️",
                        61: "Light Rain 🌧️",
                        63: "Moderate Rain 🌧️",
                        65: "Heavy Rain 🌧️",
                        71: "Snow ❄️",
                        80: "Rain Showers 🌦️",
                        81: "Moderate Showers 🌧️",
                        95: "Thunderstorm ⛈️"
                    }
                    condition = weather_codes.get(current["weather_code"], "Cloudy / Variable Conditions")
                    st.info(f"**Current Atmosphere:** {condition}")

                    # 7-day Forecast
                    st.subheader("📅 7-Day Agricultural Forecast")
                    daily = data["daily"]
                    forecast_df = pd.DataFrame({
                        "Date": daily["time"],
                        "Max Temp (°C)": daily["temperature_2m_max"],
                        "Min Temp (°C)": daily["temperature_2m_min"],
                        "Rain Probability (%)": daily["precipitation_probability_max"]
                    })
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            except requests.RequestException as e:
                st.error(f"Weather API connection timed out or failed: {e}")
            except Exception as e:
                st.error(f"Could not parse weather telemetry: {e}")

# ---------------------------------------------------------
# View 6: Farming Insights
# ---------------------------------------------------------
elif menu == "Farming Insights":
    st.title("💡 Smart Farming & Agronomy Insights")
    st.write("Best agronomic practices, integrated pest management principles, and sustainable farming guidelines.")
    st.markdown("---")

    tab1, tab2, tab3 = st.columns(3)
    with tab1:
        st.markdown("### 💧 Drip & Precision Irrigation")
        st.markdown(
            """
            - **Efficiency:** Drip irrigation reduces water wastage by up to 50% compared to flood irrigation.
            - **Timing:** Water crops during early morning or late evening to minimize evaporative loss.
            - **Sensors:** Use tensiometers or dielectric soil moisture probes at root depth.
            """
        )

    with tab2:
        st.markdown("### 🧪 4R Nutrient Stewardship")
        st.markdown(
            """
            - **Right Source:** Match fertilizer type to plant needs (e.g. nitrate vs ammonium nitrogen).
            - **Right Rate:** Base applications on soil testing results rather than generic schedules.
            - **Right Time:** Apply nutrients when crops are in peak uptake phases.
            - **Right Place:** Place fertilizer where plant roots can easily access it.
            """
        )

    with tab3:
        st.markdown("### 🛡️ Integrated Pest Defense")
        st.markdown(
            """
            - **Scouting:** Inspect fields twice weekly for pest presence and natural predators.
            - **Biological Control:** Encourage ladybugs, lacewings, and parasitoid wasps.
            - **Crop Rotation:** Rotate Solanaceae (tomatoes, potatoes) with Poaceae (cereals) or Fabaceae (legumes).
            """
        )

    st.markdown("---")
    st.markdown("### 🗓️ Seasonal Crop Calendar Overview")
    calendar_data = pd.DataFrame({
        "Cropping Season": ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"],
        "Sowing Period": ["June - July", "October - November", "March - April"],
        "Harvesting Period": ["September - October", "March - April", "May - June"],
        "Typical Crops": ["Rice, Maize, Cotton, Soybean, Groundnut", "Wheat, Mustard, Chickpea, Barley, Peas", "Watermelon, Cucumber, Muskmelon, Fodder"]
    })
    st.dataframe(calendar_data, use_container_width=True, hide_index=True)
