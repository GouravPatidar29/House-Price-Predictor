import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import seaborn as sns

# Load trained model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set Streamlit title
st.title("California Housing Price Predictor")

# Guidance Section
st.markdown("### 🧠 Input Guidance")
st.info("""
Use this guide to understand what values to enter. All values should be based on local area data.
Below is a summary of typical value ranges from the California housing dataset.
""")

try:
    housing = pd.read_csv("housing.csv")
    housing["total_bedrooms"] = housing["total_bedrooms"].fillna(housing["total_bedrooms"].median())

    st.markdown("#### 📋 Recommended Input Ranges (Based on Dataset Statistics)")
    reference_data = {
        "Feature": [
            "Median Income ($10k)", "Housing Median Age", "Total Rooms",
            "Total Bedrooms", "Population", "Households"
        ],
        "Min": [0.5, 1, 2, 1, 3, 1],
        "25%": [2.56, 18, 1447.75, 297, 787, 280],
        "Median": [3.53, 29, 2127, 435, 1166, 409],
        "75%": [4.74, 37, 3148, 643.25, 1725, 605],
        "Max": [15.0, 52, 39320, 6445, 35682, 6082],
        "Recommended Range": [
            "2.5 - 5.0", "15 - 40", "1500 - 4000",
            "300 - 700", "800 - 2000", "300 - 700"
        ]
    }
    st.table(pd.DataFrame(reference_data))

    with st.expander("🔎 Click to View Dataset Insights"):
        st.write(housing.describe())
        st.dataframe(housing.sample(3))

    st.markdown("#### 🌍 Sample Coordinates by City")
    st.table(pd.DataFrame({
        "City": ["Los Angeles", "San Francisco", "San Diego", "Sacramento", "Fresno"],
        "Latitude": [34.05, 37.77, 32.72, 38.58, 36.74],
        "Longitude": [-118.24, -122.42, -117.16, -121.49, -119.78]
    }))

    st.markdown("🔗 [Need help finding your Latitude and Longitude? Click here](https://www.latlong.net/)")

except Exception as e:
    st.warning(f"Could not load dataset summary: {e}")



with st.expander("📊 Click to View Feature Distributions"):
    tab1, tab2 = st.tabs(["📈 Histograms", "🗺️ Location Map"])

    with tab1:
        st.markdown("### Feature Histograms")
        numeric_cols = ["median_income", "housing_median_age", "total_rooms", 
                        "total_bedrooms", "population", "households"]

        for i in range(0, len(numeric_cols), 3):  # 3 plots per row
            cols = st.columns(3)
            for j, col in enumerate(numeric_cols[i:i+3]):
                with cols[j]:
                    st.markdown(f"**{col.replace('_', ' ').title()}**")
                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    sns.histplot(housing[col], bins=30, kde=False, ax=ax, color="#1f77b4")
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.grid(True, linestyle='--', alpha=0.4)
                    st.pyplot(fig)

    with tab2:
        st.markdown("### 📍 California Housing Prices by Location")
        fig, ax = plt.subplots(figsize=(8, 4))
        sc = ax.scatter(housing["longitude"], housing["latitude"],
                        c=housing["median_house_value"], cmap="coolwarm", alpha=0.6)
        plt.colorbar(sc, label="House Value")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        st.pyplot(fig)

# Define input fields
median_income = st.number_input("Median Income", min_value=0.0, step=0.1)
housing_median_age = st.number_input("Housing Median Age", min_value=1, step=1)
total_rooms = st.number_input("Total Rooms", min_value=1, step=1)
total_bedrooms = st.number_input("Total Bedrooms", min_value=1, step=1)
population = st.number_input("Population", min_value=1, step=1)
households = st.number_input("Households", min_value=1, step=1)
latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, step=0.01)
longitude = st.number_input("Longitude", min_value=-124.0, max_value=-114.0, step=0.01)

ocean_proximity = st.selectbox("Ocean Proximity", [
    "INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"
])

# Map label to encoding (from LabelEncoder used during training)
ocean_proximity_map = {
    "INLAND": 0,
    "<1H OCEAN": 1,
    "NEAR OCEAN": 2,
    "NEAR BAY": 3,
    "ISLAND": 4
}

# Convert input to model format
if st.button("Predict House Price"):
    input_data = np.array([[median_income, housing_median_age, total_rooms,
                            total_bedrooms, population, households,
                            latitude, longitude, ocean_proximity_map[ocean_proximity]]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    inr_price = prediction * 83  # Conversion from USD to INR
    st.success(f"Predicted House Price: ₹{inr_price:,.2f}")

# Model Evaluation Dashboard
st.header("📊 Model Evaluation Metrics")

if st.checkbox("Show Evaluation Metrics"):
    try:
        # Load or recreate training data here if needed (ensure consistency with actual app)
        import pandas as pd
        from sklearn.model_selection import train_test_split

        housing = pd.read_csv("housing.csv")
        housing["total_bedrooms"] = housing["total_bedrooms"].fillna(housing["total_bedrooms"].median())
        housing["ocean_proximity"] = housing["ocean_proximity"].map(ocean_proximity_map)

        X = housing.drop("median_house_value", axis=1)
        y = housing["median_house_value"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        cv_rmse = np.mean(np.sqrt(-cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)))

        st.write(f"**Train RMSE:** {train_rmse:.2f}")
        st.write(f"**Test RMSE:** {test_rmse:.2f}")
        st.write(f"**Cross-Validation RMSE:** {cv_rmse:.2f}")
        st.write(f"**Test RMSE (INR):** ₹{test_rmse * 83:,.2f}")

        # Feature importance plot
        st.subheader("🔍 Feature Importance")
        importance = model.feature_importances_
        features = X.columns

        fig, ax = plt.subplots()
        ax.barh(features, importance)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")
