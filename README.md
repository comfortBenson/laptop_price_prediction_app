💻 Laptop Price Prediction (Nigeria)
Overview:

 Buying a laptop in Nigeria can be challenging due to price volatility and varying hardware configurations. This project uses a Random Forest Regressor to provide an intelligent price estimate by analyzing factors like Brand, RAM, Storage, and CPU/GPU types, converted to Naira using current market exchange rates.The model was trained on a dataset of laptops and their specification.

 LIVE DEMO:

Project Structure:
    app.py                # Main Streamlit application
    laptop_model.pkl      # Trained Random Forest model
    encoders.pkl          # Saved LabelEncoders for categorical data
    laptops.csv           # Raw dataset
    requirements.txt      # List of dependencies
    Laptop_Analysis.ipynb # Jupyter Notebook with EDA and Model Training

 Dataset:
    Source: Laptop Price Prediction Dataset
    Columns include Brand, RAM, Storage, Storage type, CPU brand, GPU brand and Screen Size.

 Tech Stack:
    Language: Python 3.12
    Data Analysis: Pandas, NumPy
    Visualization: Matplotlib, Seaborn
    Machine Learning: Scikit-Learn (RandomForestRegressor, LabelEncoder)
    Deployment: Streamlit, GitHub

This project demonstrates end-to-end data science skills including:
    Data cleaning & feature engineering
    Exploratory Data Analysis (EDA)
    Model training & evaluation
    Deployment using Streamlit

WorkFlow:
    Data Cleaning: Handled missing values (specifically in GPU and Storage Type columns).
    ​Feature Engineering: - Extracted CPU_Brand and GPU_Brand from raw text. 
    ​Converted prices to Naira (NGN) using a standard exchange rate.
    ​Exploratory Data Analysis (EDA): Identified correlations between RAM/Storage and price.
    ​Encoding: Used LabelEncoder to transform categorical variables for production stability.
    ​Model Training: Utilized RandomForestRegressor for its ability to handle non-linear relationships in hardware specs.

Features Used for Modeling:
    Predictive Modeling: Estimates laptop prices with high accuracy using a Random Forest model.
    Interactive EDA: Visualizes market trends, brand dominance, and price distributions directly in the app.
    Currency Conversion: Automatically handles the conversion from global pricing to Naira (NGN).
    User-Friendly Interface: Built with Streamlit for a seamless sidebar-based input experience.

Model:
    Random Forest Regressor was used for price prediction.
    Evaluation Metric:
    RMSE (Root Mean Squared Error)

Installation:
    clone the repo:
    git clone <repo-url>
Install requirement:
    pip install -r requirements.txt
Run app:
    streamlit run app.py

Contact:
Built by Idongesit Benson (comfort)
GitHub: github.com/comfortBenson