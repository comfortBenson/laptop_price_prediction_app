import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

STORAGE_TYPE_COL = "Storage type"

st.set_page_config(page_title="Naira Laptop Predictor", layout="wide")

# Load Data & Model Artifacts
@st.cache_resource
def load_model():
    model = joblib.load("laptop_model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

@st.cache_data
def load_data():
    df = pd.read_csv("laptops.csv")
    df["Price_NGN"] = df["Final Price"] * 1600
    return df

model, encoders = load_model()
df = load_data()

st.title("🇳🇬 Laptop Price Predictor")
st.markdown("### Intelligent Market Value Estimation for the Nigerian Market")

# Sidebar Inputs
st.sidebar.header("Laptop Specifications")

brand = st.sidebar.selectbox("Brand", encoders["Brand"].classes_)
status = st.sidebar.selectbox("Status", encoders["Status"].classes_)
cpu_brand = st.sidebar.selectbox("CPU Brand", encoders["CPU_Brand"].classes_)
gpu_brand = st.sidebar.selectbox("GPU Brand", encoders["GPU_Brand"].classes_)
storage_type = st.sidebar.selectbox("Storage type", encoders[STORAGE_TYPE_COL].classes_)

ram = st.sidebar.slider("RAM (GB)", 4, 128, 8, step=4)
storage = st.sidebar.select_slider("Storage (GB)", options=[128, 256, 512, 1024, 2048])
screen = st.sidebar.number_input("Screen Size (Inches)", 10.0, 20.0, 15.6)

# Main Section: EDA Expander
with st.expander("📊 Explore Market Insights (EDA)"):
    col1, col2 = st.columns(2)

    with col1:
        st.write("*Price Distribution*")
        fig, ax = plt.subplots()
        sns.histplot(df["Price_NGN"], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("*Top 10 Brands by Price*")
        fig2, ax2 = plt.subplots()
        top_brands = df.groupby("Brand")["Price_NGN"].median().sort_values(ascending=False).head(10)
        sns.barplot(x=top_brands.values, y=top_brands.index, ax=ax2)
        st.pyplot(fig2)


# Prediction Logic
st.subheader("Selected Specifications")

specs_df = pd.DataFrame({
    "Feature": ["Brand", "Status", "CPU", "GPU", "RAM", "Storage", "Type", "Screen"],
    "Value": [brand, status, cpu_brand, gpu_brand, f"{ram}GB", f"{storage}GB", storage_type, f"{screen}\""]
})

st.table(specs_df.set_index("Feature"))

if st.button("Calculate Estimated Price"):
    try:
        input_dict = {
            "Brand": encoders["Brand"].transform([brand])[0],
            "Status": encoders["Status"].transform([status])[0],
            "RAM": ram,
            "Storage": storage,
            "STORAGE_TYPE_COL": encoders["STORAGE_TYPE_COL"].transform([storage_type])[0],
            "Screen": screen,
            "CPU_Brand": encoders["CPU_Brand"].transform([cpu_brand])[0],
            "GPU_Brand": encoders["GPU_Brand"].transform([gpu_brand])[0],
        }

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]

        st.success(f"### Estimated Price: ₦{prediction:,.2f}")
        st.info("Note: This is a predicted value and may vary in real market conditions.")
        st.balloons()

    except Exception:
        st.error("Error: Please try a different combination of laptop specs.")