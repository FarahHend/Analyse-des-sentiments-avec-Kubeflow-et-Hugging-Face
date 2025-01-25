import pandas as pd  # Add this import for pandas
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.model_selection import train_test_split

# Charger le dataset depuis le CSV local
dataset = pd.read_csv(r'C:\Users\hend8\Downloads\IMDB-Dataset.csv')  # Remplacez ce chemin par le chemin correct du fichier

# Vérifier l'aperçu du dataset
print(dataset.head())

# Sélectionner les colonnes de texte et d'étiquette
dataset = dataset[['review', 'sentiment']]  # Adaptez les noms des colonnes si nécessaire

# Diviser en données d'entraînement et de test
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Charger le tokenizer et le modèle pré-entraîné de Hugging Face pour l'analyse des sentiments
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Modèle pré-entraîné pour analyse des sentiments
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenisation des données d'entraînement et de test
def tokenize_function(examples):
    return tokenizer(examples["review"], padding="max_length", truncation=True, max_length=512)

# Tokeniser les ensembles de données d'entraînement et de test
train_encodings = tokenizer(list(train_data['review']), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_data['review']), truncation=True, padding=True, max_length=512)

# Créer les DataLoader pour le modèle
from torch.utils.data import DataLoader, TensorDataset

# Convertir les données en format tensor
train_inputs = torch.tensor(train_encodings['input_ids'])
train_labels = torch.tensor([1 if sentiment == 'positive' else 0 for sentiment in train_data['sentiment']])

test_inputs = torch.tensor(test_encodings['input_ids'])
test_labels = torch.tensor([1 if sentiment == 'positive' else 0 for sentiment in test_data['sentiment']])

# Créer les DataLoader
train_data = TensorDataset(train_inputs, train_labels)
test_data = TensorDataset(test_inputs, test_labels)

train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)
