# ChronoRoute-Amazon-Delivery-Time-Prediction
Machine Learning model (RFR/XGBoost) to predict sub-minute delivery ETAs using dynamic features (traffic, distance, weather). Deployed via Streamlit for real-time forecasting. 


# 🚚 ChronoRoute: Real-Time Delivery Time Prediction

## Project Overview

**ChronoRoute** is a machine learning solution designed to predict the Estimated Time of Arrival (ETA) for food and package deliveries with high accuracy. This project leverages an optimized **XGBoost/Random Forest Regressor** model, trained on extensive real-world logistical data, and deploys it as an interactive web application using **Streamlit**.

The model accounts for crucial, dynamic variables that impact delivery time, including:
* **Logistics:** Agent age, rating, and order preparation time.
* **Environmental Factors:** Real-time weather and traffic conditions.
* **Geospatial Data:** Calculated distance (using Haversine formula) and delivery area type.
* **Temporal Data:** Time of day (e.g., Morning, Evening rush).

***

## 🚀 Getting Started

Follow these steps to set up and run the prediction application locally.

### Prerequisites

You need Python 3.8+ installed. The following libraries are required:

```bash
pip install streamlit pandas numpy scikit-learn joblib
````

### Files Required

Ensure the following files are in your project directory:

1.  `app.py` (The Streamlit application code)
2.  `best_delivery_predictor.joblib` (The serialized trained model)
3.  `bg.jpg` (The custom background image)

### Execution

1.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
2.  **Access:** The app will automatically open in your web browser at `http://localhost:8501`.

-----

## 🧠 Machine Learning Details

### Model

The final model used for deployment is an optimized **Random Forest Regressor** (or XGBoost, based on your training), selected after extensive hyperparameter tuning (e.g., via Randomized Search CV).

### Key Features & Engineering

The model's prediction relies on 37 features derived from the raw input. Key engineered features include:

  * **Distance:** Calculated using the **Haversine formula** based on store and drop-off coordinates.
  * **Temporal Encoding:** Order time is decomposed into `Order_Hour`, `Order\_Minute`, and `Time\_of\_Day` (Morning, Afternoon, Evening, Night).
  * **Feature Scaling:** All numerical features (e.g., Agent Age, Distance) are standardized using the **Z-score method**.
  * **Categorical Encoding:** **One-Hot Encoding (OHE)** is applied to all nominal features (Weather, Traffic, Vehicle, etc.), with a baseline category dropped to prevent multicollinearity.

-----

## 📈 MLflow Tracking

Model development and performance tracking were managed using **MLflow**. The provided `log\_model.py` script demonstrates how the final, optimized model's metrics (`RMSE`, `R2\_Score`), parameters, and artifact file were logged for reproducibility.

-----

## 🛠️ Customization

### Background & Styling

The background image (`bg.jpg`) and all application styling (dark background, white labels, black input text) are managed by injecting custom CSS within the `set_background` function in `app.py`. To change the background, simply replace `bg.jpg` with your image.

```
```
