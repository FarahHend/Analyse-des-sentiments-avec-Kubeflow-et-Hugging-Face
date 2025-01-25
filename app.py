from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load a pre-trained fine-tuned sentiment analysis model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # A model already fine-tuned for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Initialize FastAPI
app = FastAPI()

# Define request model for input
class Review(BaseModel):
    text: str

# Prediction function with added debug prints
def predict_sentiment(text):
    # Tokenize input text and ensure padding and truncation are applied correctly
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)  # Pass inputs through the model
    logits = outputs.logits  # Get the logits (raw predictions)

    # Debugging: Print the logits to check if there is a significant difference between them
    print(f"Logits: {logits}")
    
    # Get the predicted class
    pred = torch.argmax(logits, dim=1)  # Argmax to get the index of the maximum value
    sentiment = "positive" if pred == 1 else "negative"  # Interpret prediction
    return sentiment

# Define route for inference
@app.post("/predict")
def predict(review: Review):
    sentiment = predict_sentiment(review.text)
    return {"sentiment": sentiment}

# Define a GET route for the root URL
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API. Use /predict to analyze sentiment."}
