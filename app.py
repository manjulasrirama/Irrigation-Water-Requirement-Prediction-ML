import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Irrigation Water Prediction",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL AND DATASET
# --------------------------------------------------

model = joblib.load("models/irrigation_model.pkl")

df = pd.read_csv(
    "dataset/irrigation_prediction.csv"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🌱 Irrigation Water Requirement Prediction")

st.write(
    "Enter the soil, weather, crop and irrigation details "
    "to predict the irrigation requirement."
)

st.divider()


# --------------------------------------------------
# SOIL INFORMATION
# --------------------------------------------------

st.header("🌍 Soil Information")

col1, col2 = st.columns(2)

with col1:

    soil_type = st.selectbox(
        "Soil Type",
        sorted(df["Soil_Type"].unique())
    )

    soil_ph = st.number_input(
        "Soil pH",
        value=float(df["Soil_pH"].median())
    )

    soil_moisture = st.number_input(
        "Soil Moisture",
        value=float(df["Soil_Moisture"].median())
    )


with col2:

    organic_carbon = st.number_input(
        "Organic Carbon",
        value=float(df["Organic_Carbon"].median())
    )

    electrical_conductivity = st.number_input(
        "Electrical Conductivity",
        value=float(
            df["Electrical_Conductivity"].median()
        )
    )


st.divider()


# --------------------------------------------------
# WEATHER INFORMATION
# --------------------------------------------------

st.header("🌤️ Weather Information")

col1, col2, col3 = st.columns(3)

with col1:

    temperature = st.number_input(
        "Temperature (°C)",
        value=float(
            df["Temperature_C"].median()
        )
    )


with col2:

    humidity = st.number_input(
        "Humidity",
        value=float(
            df["Humidity"].median()
        )
    )


with col3:

    rainfall = st.number_input(
        "Rainfall (mm)",
        value=float(
            df["Rainfall_mm"].median()
        )
    )


col1, col2 = st.columns(2)

with col1:

    sunlight = st.number_input(
        "Sunlight Hours",
        value=float(
            df["Sunlight_Hours"].median()
        )
    )


with col2:

    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        value=float(
            df["Wind_Speed_kmh"].median()
        )
    )


st.divider()


# --------------------------------------------------
# CROP INFORMATION
# --------------------------------------------------

st.header("🌾 Crop Information")

col1, col2, col3 = st.columns(3)

with col1:

    crop_type = st.selectbox(
        "Crop Type",
        sorted(
            df["Crop_Type"].unique()
        )
    )


with col2:

    crop_growth_stage = st.selectbox(
        "Crop Growth Stage",
        sorted(
            df["Crop_Growth_Stage"].unique()
        )
    )


with col3:

    season = st.selectbox(
        "Season",
        sorted(
            df["Season"].unique()
        )
    )


st.divider()


# --------------------------------------------------
# IRRIGATION INFORMATION
# --------------------------------------------------

st.header("💧 Irrigation Information")

col1, col2 = st.columns(2)

with col1:

    irrigation_type = st.selectbox(
        "Irrigation Type",
        sorted(
            df["Irrigation_Type"].unique()
        )
    )

    water_source = st.selectbox(
        "Water Source",
        sorted(
            df["Water_Source"].unique()
        )
    )

    field_area = st.number_input(
        "Field Area (hectare)",
        value=float(
            df["Field_Area_hectare"].median()
        )
    )


with col2:

    mulching_used = st.selectbox(
        "Mulching Used",
        sorted(
            df["Mulching_Used"].unique()
        )
    )

    previous_irrigation = st.number_input(
        "Previous Irrigation (mm)",
        value=float(
            df["Previous_Irrigation_mm"].median()
        )
    )

    region = st.selectbox(
        "Region",
        sorted(
            df["Region"].unique()
        )
    )


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔮 Predict Irrigation Need",
    use_container_width=True
):

    try:

        # --------------------------------------------------
        # CREATE INPUT DATA
        # --------------------------------------------------

        input_data = pd.DataFrame({

            "Soil_Type": [soil_type],

            "Soil_pH": [soil_ph],

            "Soil_Moisture": [
                soil_moisture
            ],

            "Organic_Carbon": [
                organic_carbon
            ],

            "Electrical_Conductivity": [
                electrical_conductivity
            ],

            "Temperature_C": [
                temperature
            ],

            "Humidity": [
                humidity
            ],

            "Rainfall_mm": [
                rainfall
            ],

            "Sunlight_Hours": [
                sunlight
            ],

            "Wind_Speed_kmh": [
                wind_speed
            ],

            "Crop_Type": [
                crop_type
            ],

            "Crop_Growth_Stage": [
                crop_growth_stage
            ],

            "Season": [
                season
            ],

            "Irrigation_Type": [
                irrigation_type
            ],

            "Water_Source": [
                water_source
            ],

            "Field_Area_hectare": [
                field_area
            ],

            "Mulching_Used": [
                mulching_used
            ],

            "Previous_Irrigation_mm": [
                previous_irrigation
            ],

            "Region": [
                region
            ]
        })


        # --------------------------------------------------
        # MAKE PREDICTION
        # --------------------------------------------------

        prediction = model.predict(
            input_data
        )

        result = prediction[0]


        # --------------------------------------------------
        # DISPLAY PREDICTION
        # --------------------------------------------------

        st.subheader("🌱 Prediction Result")

        if result == "Low":

            st.success(
                "🌱 Irrigation Requirement: **LOW**"
            )

        elif result == "Medium":

            st.warning(
                "🌱 Irrigation Requirement: **MEDIUM**"
            )

        elif result == "High":

            st.error(
                "🌱 Irrigation Requirement: **HIGH**"
            )

        else:

            st.info(
                f"🌱 Irrigation Requirement: **{result}**"
            )


        # --------------------------------------------------
        # PREDICTION PROBABILITIES
        # --------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = model.classes_


            probability_df = pd.DataFrame({

                "Irrigation Level": classes,

                "Probability (%)": (
                    probabilities * 100
                ).round(2)

            })


            st.subheader(
                "📊 Prediction Probabilities"
            )


            # Display probabilities

            for i, row in probability_df.iterrows():

                level = row["Irrigation Level"]

                probability = row["Probability (%)"]


                st.write(
                    f"**{level}: {probability:.2f}%**"
                )


                st.progress(
                    int(probability)
                )


            # Display table

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


            st.info(
                "The probabilities represent the model's "
                "confidence for each irrigation requirement level."
            )


    except Exception as e:

        st.error(
            f"⚠️ Prediction error: {e}"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Based Irrigation Water Requirement Prediction"
)