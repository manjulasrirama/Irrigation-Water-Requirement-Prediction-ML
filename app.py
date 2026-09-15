import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Irrigation Water Prediction",
    page_icon="🌱",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND DATASET
# =========================================================

model = joblib.load("models/irrigation_model.pkl")

df = pd.read_csv("dataset/irrigation_prediction.csv")


# =========================================================
# TITLE
# =========================================================

st.title("🌱 Irrigation Water Requirement Prediction")

st.write(
    "Enter the soil, weather, crop and irrigation details "
    "to predict the irrigation requirement."
)

st.divider()


# =========================================================
# SOIL INFORMATION
# =========================================================

st.header("🌍 Soil Information")

col1, col2 = st.columns(2)


with col1:

    soil_type = st.selectbox(
        "Soil Type",
        sorted(df["Soil_Type"].unique())
    )

    soil_ph = st.text_input(
        "Soil pH",
        value=str(round(df["Soil_pH"].median(), 2))
    )

    soil_moisture = st.text_input(
        "Soil Moisture",
        value=str(round(df["Soil_Moisture"].median(), 2))
    )


with col2:

    organic_carbon = st.text_input(
        "Organic Carbon",
        value=str(round(df["Organic_Carbon"].median(), 2))
    )

    electrical_conductivity = st.text_input(
        "Electrical Conductivity",
        value=str(round(df["Electrical_Conductivity"].median(), 2))
    )


st.divider()


# =========================================================
# WEATHER INFORMATION
# =========================================================

st.header("🌤️ Weather Information")

col1, col2, col3 = st.columns(3)


with col1:

    temperature = st.text_input(
        "Temperature (°C)",
        value=str(round(df["Temperature_C"].median(), 2))
    )


with col2:

    humidity = st.text_input(
        "Humidity",
        value=str(round(df["Humidity"].median(), 2))
    )


with col3:

    rainfall = st.text_input(
        "Rainfall (mm)",
        value=str(round(df["Rainfall_mm"].median(), 2))
    )


col1, col2 = st.columns(2)


with col1:

    sunlight = st.text_input(
        "Sunlight Hours",
        value=str(round(df["Sunlight_Hours"].median(), 2))
    )


with col2:

    wind_speed = st.text_input(
        "Wind Speed (km/h)",
        value=str(round(df["Wind_Speed_kmh"].median(), 2))
    )


st.divider()


# =========================================================
# CROP INFORMATION
# =========================================================

st.header("🌾 Crop Information")

col1, col2, col3 = st.columns(3)


with col1:

    crop_type = st.selectbox(
        "Crop Type",
        sorted(df["Crop_Type"].unique())
    )


with col2:

    crop_growth_stage = st.selectbox(
        "Crop Growth Stage",
        sorted(df["Crop_Growth_Stage"].unique())
    )


with col3:

    season = st.selectbox(
        "Season",
        sorted(df["Season"].unique())
    )


st.divider()


# =========================================================
# IRRIGATION INFORMATION
# =========================================================

st.header("💧 Irrigation Information")

col1, col2 = st.columns(2)


with col1:

    irrigation_type = st.selectbox(
        "Irrigation Type",
        sorted(df["Irrigation_Type"].unique())
    )

    water_source = st.selectbox(
        "Water Source",
        sorted(df["Water_Source"].unique())
    )

    field_area = st.text_input(
        "Field Area (hectare)",
        value=str(round(df["Field_Area_hectare"].median(), 2))
    )


with col2:

    mulching_used = st.selectbox(
        "Mulching Used",
        sorted(df["Mulching_Used"].unique())
    )

    previous_irrigation = st.text_input(
        "Previous Irrigation (mm)",
        value=str(round(df["Previous_Irrigation_mm"].median(), 2))
    )

    region = st.selectbox(
        "Region",
        sorted(df["Region"].unique())
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔮 Predict Irrigation Need",
    use_container_width=True
):

    try:

        # Convert manually entered values to numbers

        soil_ph_value = float(soil_ph)

        soil_moisture_value = float(soil_moisture)

        organic_carbon_value = float(organic_carbon)

        electrical_conductivity_value = float(
            electrical_conductivity
        )

        temperature_value = float(temperature)

        humidity_value = float(humidity)

        rainfall_value = float(rainfall)

        sunlight_value = float(sunlight)

        wind_speed_value = float(wind_speed)

        field_area_value = float(field_area)

        previous_irrigation_value = float(
            previous_irrigation
        )


        # =================================================
        # CREATE INPUT DATAFRAME
        # =================================================

        input_data = pd.DataFrame({

            "Soil_Type": [soil_type],

            "Soil_pH": [soil_ph_value],

            "Soil_Moisture": [soil_moisture_value],

            "Organic_Carbon": [organic_carbon_value],

            "Electrical_Conductivity": [
                electrical_conductivity_value
            ],

            "Temperature_C": [temperature_value],

            "Humidity": [humidity_value],

            "Rainfall_mm": [rainfall_value],

            "Sunlight_Hours": [sunlight_value],

            "Wind_Speed_kmh": [wind_speed_value],

            "Crop_Type": [crop_type],

            "Crop_Growth_Stage": [crop_growth_stage],

            "Season": [season],

            "Irrigation_Type": [irrigation_type],

            "Water_Source": [water_source],

            "Field_Area_hectare": [
                field_area_value
            ],

            "Mulching_Used": [mulching_used],

            "Previous_Irrigation_mm": [
                previous_irrigation_value
            ],

            "Region": [region]
        })


        # =================================================
        # MAKE PREDICTION
        # =================================================

        prediction = model.predict(input_data)

        result = prediction[0]


        # =================================================
        # DISPLAY RESULT
        # =================================================

        st.success(
            f"🌱 Irrigation Requirement: **{result}**"
        )


    except ValueError:

        st.error(
            "⚠️ Please enter valid numbers in all "
            "numeric fields."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Machine Learning Based Irrigation Water Requirement Prediction"
)