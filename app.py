import streamlit as st
import joblib
import numpy as np



# Set page configuration
st.set_page_config(page_title="Human Stress Level Detection", page_icon="🧠", layout="centered")

# CSS to inject contained in a string
css = """
<style>
body {
    background-color: black;
    color: white;
}
</style>
"""

# Inject CSS with markdown
st.markdown(css, unsafe_allow_html=True)




# Load the trained model and scaler
model = joblib.load('human_stress_detection.pkl')
scaler = joblib.load('scaler.pkl')


st.title('Human Stress Level Detection')

st.write("""
This app predicts your stress level based on various physiological indicators.
""")

# Collect user input
snoring_rate = st.number_input('Snoring Rate', min_value=0.0, max_value=100.0, value=50.0)
respiration_rate = st.number_input('Respiration Rate', min_value=0.0, max_value=100.0, value=20.0)
body_temperature = st.number_input('Body Temperature', min_value=0.0, max_value=100.0, value=36.5)
limb_movement = st.number_input('Limb Movement', min_value=0.0, max_value=100.0, value=10.0)
blood_oxygen = st.number_input('Blood Oxygen', min_value=0.0, max_value=100.0, value=95.0)
eye_movement = st.number_input('Eye Movement', min_value=0.0, max_value=100.0, value=1.0)
sleeping_hours = st.number_input('Sleeping Hours', min_value=0.0, max_value=24.0, value=7.0)
heart_rate = st.number_input('Heart Rate', min_value=0.0, max_value=200.0, value=70.0)

# Create a feature array
features = np.array([[
    snoring_rate, respiration_rate, body_temperature, limb_movement,
    blood_oxygen, eye_movement, sleeping_hours, heart_rate
]])

# Scale the input features
features_scaled = scaler.transform(features)

# Predict stress level
if st.button('Predict Stress Level'):
    prediction = model.predict(features_scaled)
    stress_level = int(prediction[0])
    st.write(f'The predicted stress level is: {stress_level}')
    if stress_level == 1:
        print("Stress level=1")
        st.success("The person has low stress level 🙂")
    elif stress_level == 2:
        print("Stress level=2")
        st.warning("The person has medium stress level 😐")
    elif stress_level == 3:
        print("Stress level=3")
        st.error("The person has high stress level! 😞")
    elif stress_level == 4:
        print("Stress level=4")
        st.error("The person has very high stress level!! 😫")
    else:
        print("Stress level=0")
        st.success("The person is stress free and calm 😄")
