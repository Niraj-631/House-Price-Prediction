import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# Load the trained model pipeline
# -----------------------------
if os.path.exists("RidgeModel.pkl"):
    with open("RidgeModel.pkl", "rb") as f:
        model = pickle.load(f)
else:
    st.error("🚨 Model file 'RidgeModel.pkl' not found! Please upload it to the app directory.")
    st.stop()


# -----------------------------
# Extract location list from trained encoder
# -----------------------------
try:
    ohe_categories = model.named_steps['columntransformer'].transformers_[0][1].categories_[0]
    locations_list = sorted(ohe_categories)
except:
    locations_list = ['Thanisandra', 'Whitefield', 'Electronic City']

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏠 House Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict house prices based on location, sqft, bathrooms and BHK. 💰</p>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# User inputs with columns
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    location = st.selectbox("📍 Location", locations_list)
with col2:
    sqft = st.number_input("📐 Total Sqft", min_value=300, max_value=12000, step=50, value=1000)
with col3:
    bath = st.slider("🛁 Bathrooms", 1, 5, 2)
with col4:
    bhk = st.slider("🛏 BHK", 1, 5, 3)

# Prepare input dataframe
input_df = pd.DataFrame({
    'location': [location],
    'total_sqft': [sqft],
    'bath': [bath],
    'bhk': [bhk]
})

st.markdown("### 🔎 Input Summary")
st.table(input_df)

# -----------------------------
# Predict button with metric
# -----------------------------
if st.button("💰 Predict Price"):
    predicted_price = model.predict(input_df)[0]
    st.success(f"✅ Estimated House Price: ₹ {predicted_price:,.2f}")

    colA, colB, colC = st.columns(3)
    colA.metric("📍 Location", location)
    colB.metric("🏠 Area (sqft)", f"{sqft}")
    colC.metric("💸 Predicted Price (₹)", f"{predicted_price:,.0f}")

    # -----------------------------
    # Log to CSV
    # -----------------------------
    log_df = input_df.copy()
    log_df['predicted_price'] = predicted_price
    if os.path.exists("history.csv"):
        history = pd.read_csv("history.csv")
        history = pd.concat([history, log_df], ignore_index=True)
    else:
        history = log_df
    history.to_csv("history.csv", index=False)

    # -----------------------------
    # Show more insights
    # -----------------------------
    with st.expander("📊 View Prediction History & Stats"):
        st.dataframe(history.tail(10))

        # Bar chart by location
        location_avg = history.groupby('location')['predicted_price'].mean().reset_index()
        st.bar_chart(location_avg.set_index('location'))

    st.balloons()
