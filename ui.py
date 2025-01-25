# ui.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Placeholder for experiment tracking (metrics, hyperparameters)
st.title("Sentiment Analysis Experiment Tracker")

st.subheader("Model Metrics")

# Define file path
file_path = "C:/Users/hend8/Desktop/sentiment_model/metrics_log.csv"

# Check if the file exists and is not empty
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    metrics_data = pd.read_csv(file_path)
else:
    # If the file is empty or doesn't exist, create a new one with headers
    st.warning(f"'{file_path}' is empty or doesn't exist, creating a new file with headers.")
    metrics_data = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])
    metrics_data.to_csv(file_path, index=False)  # Create the empty CSV file

st.write(metrics_data)

# Show training/validation loss curve
st.subheader("Training/Validation Loss Curve")

if not metrics_data.empty:
    loss_data = metrics_data[['epoch', 'train_loss', 'val_loss']]
    plt.plot(loss_data['epoch'], loss_data['train_loss'], label='Train Loss')
    plt.plot(loss_data['epoch'], loss_data['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)
else:
    st.info("No data available for plotting loss curve.")

# You can add more visualization for accuracy, hyperparameters, etc.
