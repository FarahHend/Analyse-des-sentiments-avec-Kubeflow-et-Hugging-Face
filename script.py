import streamlit as st
import requests

# Streamlit UI
st.title("Sentiment Analysis Web App")
st.write("Enter your text below to analyze its sentiment.")

# User input
user_input = st.text_area("Text to Analyze")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        response = requests.post("http://127.0.0.1:8000/predict", json={"text": user_input})
        if response.status_code == 200:
            result = response.json()
            st.success(f"Sentiment: {result['sentiment']}")
        else:
            st.error("Failed to analyze sentiment. Please check the server.")
    else:
        st.warning("Please enter text for analysis.")
