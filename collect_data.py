import pandas as pd
from sklearn.model_selection import train_test_split

# Charger le dataset depuis le fichier CSV local
dataset = pd.read_csv(r'C:\Users\hend8\Downloads\IMDB-Dataset.csv')  # Assurez-vous de remplacer ce chemin par le chemin correct du fichier

# Aperçu des données
print("Aperçu des données :")
print(dataset.head())

# Diviser les données en train et test (80% train, 20% test)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Sauvegarder les données de train et test dans des fichiers CSV
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

# Aperçu des données après division
print("Données de train :")
print(train_data.head())

print("Données de test :")
print(test_data.head())
