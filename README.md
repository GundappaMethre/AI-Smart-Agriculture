# 🌱 AI Smart Agriculture Platform

An end-to-end intelligent agriculture platform designed to empower farmers, agronomists, and researchers with AI-driven insights for precision farming.

Deployed with [Streamlit](https://streamlit.io/), this system provides instant crop recommendations, deep learning tomato leaf disease detection, live meteorological weather telemetry, and soil health monitoring.

---

## 🌟 Key Features

1. **🌾 AI Crop Recommendation**:
   - Uses an optimized **Random Forest Classifier** trained on agricultural soil and environmental datasets.
   - Evaluates Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$), soil pH, ambient temperature, relative humidity, and rainfall to suggest the highest-yielding crop.
   - Includes quick-load soil profile presets for common crops (Rice, Cotton, Legumes, Maize).

2. **🍅 Deep Learning Plant Disease Diagnosis**:
   - Built on a **MobileNetV2** Convolutional Neural Network trained on the PlantVillage dataset.
   - Diagnoses 10 tomato plant health classes: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus (TYLCV), Mosaic Virus (ToMV), and Healthy foliage.
   - Outputs confidence scores, symptom breakdowns, and recommended agricultural treatments / remedies.

3. **🪴 Soil Health & Moisture Monitoring**:
   - Interactive diagnostic assessment of soil pH, moisture content, and N-P-K nutrient availability.
   - Provides agronomic remediation checklists (e.g., lime or gypsum application, organic compost recommendations).

4. **🌦️ Real-Time Meteorological Intelligence**:
   - Integrated with the **Open-Meteo API** (zero API keys required, global coverage).
   - Instant geocoding for any city or village, reporting ambient temperature, feels-like temperature, humidity, rainfall, wind velocity, and an automated 7-day weather forecast.

5. **💡 Agronomy Insights & Seasonal Guides**:
   - 4R Nutrient Stewardship guidelines, precision drip irrigation tips, integrated pest defense (IPM), and seasonal cropping calendars (Kharif, Rabi, Zaid).

---

## 📁 Project Architecture

```text
AI-Smart-Agriculture/
├── .streamlit/
│   └── config.toml                  # Streamlit UI theme and server configuration
├── models/
│   ├── class_names.json             # 10 tomato disease class labels
│   ├── crop_disease_model.keras     # Trained MobileNetV2 disease model (~9.7 MB)
│   └── crop_recommendation_model.pkl# Trained Random Forest crop model (~7.4 MB)
├── data/
│   └── crops.csv                    # Dataset used for crop recommendation model training
├── app.py                           # Main Streamlit web application
├── requirements.txt                 # Pinned dependencies for Streamlit Cloud deployment
├── .python-version                  # Specifies Python 3.11 runtime
├── .gitignore                       # Git ignore rules for venvs, large data, and secrets
├── train_disease.py                 # Disease classification model training script
├── train_recommendation.py          # Crop recommendation model training script
└── README.md                        # Documentation
```

---

## 🚀 How to Deploy on Streamlit Community Cloud

Streamlit Community Cloud allows free, one-click hosting for Streamlit applications directly linked to GitHub.

### Step 1: Initialize Git and Push to GitHub

In your terminal / PowerShell, navigate to the project directory:

```bash
git init
git add .
git commit -m "Initial commit for Streamlit Cloud deployment"
```

Create a new repository on [GitHub](https://github.com/new) (e.g. `AI-Smart-Agriculture`), then link and push:

```bash
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/AI-Smart-Agriculture.git
git push -u origin main
```

*(Note: The model files in `models/` total ~17 MB, which is well within GitHub's 100 MB limit, so Git LFS is not required).*

### Step 2: Deploy on Streamlit Community Cloud

1. Visit **[share.streamlit.io](https://share.streamlit.io/)** (or [streamlit.io/cloud](https://streamlit.io/cloud)) and sign in with your GitHub account.
2. Click **"New app"** (or **"Create app"**).
3. Fill in the deployment settings:
   - **Repository:** `<YOUR_GITHUB_USERNAME>/AI-Smart-Agriculture`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** Customize your app's subdomain (e.g., `ai-smart-agriculture.streamlit.app`)
4. *(Optional / Recommended)* Click **"Advanced settings..."** and confirm the Python version is set to **`3.11`**.
5. Click **"Deploy!"**.

Streamlit Cloud will install `requirements.txt`, load the models, and launch your live public web application with HTTPS. Any future `git push` to `main` will automatically trigger a live update!

---

## 💻 Running Locally

### Prerequisites
- Python 3.11 recommended (Python 3.10 – 3.11 are fully supported by TensorFlow 2.16)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<YOUR_GITHUB_USERNAME>/AI-Smart-Agriculture.git
   cd AI-Smart-Agriculture
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Technologies Used

- **Web Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
- **Deep Learning:** [TensorFlow / Keras 3](https://www.tensorflow.org/) (MobileNetV2)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Image Processing:** [Pillow (PIL)](https://python-pillow.org/)
- **Weather API:** [Open-Meteo Global Meteorological API](https://open-meteo.com/)
