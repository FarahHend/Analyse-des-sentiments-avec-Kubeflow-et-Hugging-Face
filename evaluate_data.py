from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the saved model weights
model.load_state_dict(torch.load('C:/Users/hend8/Desktop/sentiment_model/trained_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load and prepare the test dataset
dataset = pd.read_csv(r'C:\Users\hend8\Downloads\IMDB-Dataset.csv')  # Same dataset or a test dataset
dataset = dataset[['review', 'sentiment']]
test_data = dataset.iloc[:10]  # Adjust size of test data as needed

# Tokenize the test data
max_length = 128  # Same max length used in training
test_encodings = tokenizer(list(test_data['review']), truncation=True, padding=True, max_length=max_length, return_attention_mask=True)

# Convert to tensors
test_inputs = torch.tensor(test_encodings['input_ids'])
test_attention_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor([1 if sentiment == 'positive' else 0 for sentiment in test_data['sentiment']])

# Create DataLoader for test data
test_dataset = TensorDataset(test_inputs, test_attention_masks, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Function to evaluate the model
def evaluate_model(model, test_dataloader):
    predictions, true_labels = [], []
    
    with torch.no_grad():  # Turn off gradients to speed up inference
        for batch in test_dataloader:
            input_ids, attention_masks, labels = batch

            # Perform forward pass
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits

            # Get predictions (class 0 or 1)
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.numpy())  # Append predictions
            true_labels.extend(labels.numpy())  # Append true labels

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Test accuracy: {accuracy}")

# Evaluate the model on the test data
evaluate_model(model, test_dataloader)