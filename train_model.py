import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

# Charger le dataset depuis le CSV local
dataset = pd.read_csv(r'C:\Users\hend8\Downloads\IMDB-Dataset.csv')

# Prétraiter les données (sélectionner les colonnes, etc.)
dataset = dataset[['review', 'sentiment']]
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Charger le tokenizer et le modèle pré-entraîné
model_name = "bert-base-uncased"  # Using a smaller BERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Select only the first 10 samples from the training and testing sets
train_data = train_data.iloc[:1000]
test_data = test_data.iloc[:1000]

# Tokenisation des données avec un max_length réduit pour accélérer l'entraînement
max_length = 128  # Reduced sequence length
train_encodings = tokenizer(list(train_data['review']), truncation=True, padding=True, max_length=max_length, return_attention_mask=True)
test_encodings = tokenizer(list(test_data['review']), truncation=True, padding=True, max_length=max_length, return_attention_mask=True)

# Convertir en tensors
train_inputs = torch.tensor(train_encodings['input_ids'])
train_attention_masks = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor([1 if sentiment == 'positive' else 0 for sentiment in train_data['sentiment']])

test_inputs = torch.tensor(test_encodings['input_ids'])
test_attention_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor([1 if sentiment == 'positive' else 0 for sentiment in test_data['sentiment']])

# Create DataLoader with attention mask
train_data = TensorDataset(train_inputs, train_attention_masks, train_labels)
test_data = TensorDataset(test_inputs, test_attention_masks, test_labels)

train_dataloader = DataLoader(train_data, batch_size=2, shuffle=False)  # Small batch size since only 10 samples
test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False)

# Définir l'optimiseur
optimizer = AdamW(model.parameters(), lr=5e-5)

# Set up learning rate scheduler
num_epochs = 2  # Reduced the number of epochs to 2
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Gradient clipping value
max_grad_norm = 1.0  # Clip gradients to this value

# Définir une fonction pour entraîner le modèle
def train_model(model, train_dataloader, optimizer, scheduler):
    model.train()  # Passer le modèle en mode entraînement
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_masks, labels = batch
        optimizer.zero_grad()  # Remettre à zéro les gradients

        outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss  # Extraire la perte

        loss.backward()  # Backward pass
        optimizer.step()  # Optimizer step

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Training loss: {avg_loss}")

    scheduler.step()  # Update learning rate scheduler

# Entraîner le modèle
for epoch in range(num_epochs):  # Entraînement sur 2 époques
    print(f"Epoch {epoch + 1}")
    train_model(model, train_dataloader, optimizer, scheduler)

# Define the save path
# Define the save path (using a simpler folder name)
save_path = "C:/Users/hend8/Desktop/sentiment_model"

# Check if the directory exists, if not, create it
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save the model's state dictionary
model_save_path = os.path.join(save_path, "trained_model.pth")
torch.save(model.state_dict(), model_save_path)

# Optionally save the tokenizer (if needed)
tokenizer_save_path = os.path.join(save_path, "trained_tokenizer")
tokenizer.save_pretrained(tokenizer_save_path)
