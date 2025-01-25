from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define the prediction function
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1)
    sentiment = "positive" if pred == 1 else "negative"
    return sentiment

# Test with a new review
new_review = "This product is terrible, I regret buying it."
prediction = predict_sentiment(new_review, model, tokenizer)
print(f"Sentiment de la critique : {prediction}")
