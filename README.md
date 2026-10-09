Irrigation Water Requirement Prediction
Machine Learning Project — README
This project predicts an agricultural field's irrigation requirement from soil, weather, crop, and irrigation-related information. The
output is classified as Low, Medium, or High. A Streamlit web application accepts user inputs and displays the prediction with
class probabilities.
Objectives
• Predict irrigation requirement using machine learning.
• Explore soil, weather, crop, and irrigation features.
• Compare classification algorithms and tune a Random Forest model.
• Display predictions and class probabilities through a Streamlit interface.
Dataset
The project dataset is reported to contain 10,000 records, 19 input features, and the target column Irrigation_Need. Target
classes: Low, Medium, and High.
Feature groups: Soil (Soil_Type, Soil_pH, Soil_Moisture, Organic_Carbon, Electrical_Conductivity); weather (Temperature_C,
Humidity, Rainfall_mm, Sunlight_Hours, Wind_Speed_kmh); crop (Crop_Type, Crop_Growth_Stage, Season); irrigation
(Irrigation_Type, Water_Source, Previous_Irrigation_mm); other field details (Field_Area_hectare, Mulching_Used, Region).
Technologies
Python, Pandas, NumPy, Scikit-learn, Joblib, Streamlit, Matplotlib, Seaborn, Jupyter Notebook, Git, and GitHub.
Workflow
1. Load and understand the dataset.
2. Perform exploratory data analysis.
3. Preprocess numerical and categorical features.
4. Apply One-Hot Encoding to categorical features.
5. Split data into training and testing sets.
6. Train and evaluate classification models.
7. Tune the Random Forest pipeline.
8. Save the trained pipeline with Joblib.
9. Use Streamlit to display predictions and probabilities.
Models and Reported Test Accuracy
Model Test accuracy
Logistic Regression 82.60%
Decision Tree Classifier 99.60%
Random Forest Classifier 97.35%
Tuned Random Forest Classifier 96.95%
The Decision Tree achieved the highest test accuracy in the reported experiment. The tuned Random Forest pipeline was used in
the final Streamlit application. Test accuracy does not guarantee performance on new real-world agricultural data.
Run Locally
1. Clone the repository:
git clone https://github.com/manjulasrirama/Irrigation-Water-Requirement-Prediction-ML.git
cd Irrigation-Water-Requirement-Prediction-ML
2. Create and activate a virtual environment (Windows):
python -m venv venv
venv\Scripts\activate
3. Install dependencies and start the app:
pip install -r requirements.txt
streamlit run app.py
The saved pipeline should be available at models/irrigation_model.pkl. Include this file, the dataset, app.py, prediction.py, and
requirements.txt in the repository.
Project Structure
Irrigation-Water-Requirement-Prediction-ML/
■■■ dataset/irrigation_prediction.csv
■■■ models/irrigation_model.pkl
■■■ app.py
■■■ prediction.py
■■■ irrigation_prediction.ipynb
■■■ requirements.txt
■■■ README.md
Future Enhancements
Potential improvements include live weather data, soil-moisture sensors, IoT-based monitoring, and evaluation using additional
real-world agricultural datasets.
References
Scikit-learn: https://scikit-learn.org/stable/
Streamlit: https://docs.streamlit.io/
Pandas: https://pandas.pydata.org/docs/
NumPy: https://numpy.org/doc/
Joblib: https://joblib.readthedocs.io/
Academic project developed using Machine Learning and Streamlit.
