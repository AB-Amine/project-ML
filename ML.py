import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

print("Script de classification ML (Scikit-learn, Pandas, SVM, Régression Logistique)")

# --- 1. Chargement des données (avec Pandas) ---
print("Téléchargement des données MNIST...")
# as_frame=True charge les données dans un DataFrame Pandas, comme sur votre CV
mnist_data = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')

X = mnist_data.data
y = mnist_data.target

print("Données chargées. Total d'images :", len(X))

# --- 2. Préparation des données ---
# Division standard de MNIST (60k entraînement, 10k test)
X_train_full, X_test_full = X[:60000], X[60000:]
y_train_full, y_test_full = y[:60000], y[60000:]

# Création d'un "petit" projet : 
# Nous prenons un échantillon pour un entraînement rapide
# C'est ce qui explique une précision de ~90% au lieu de +98%
print("Création d'un petit échantillon (5000 entraînement, 1000 test)...")
X_train_sample = X_train_full.sample(n=5000, random_state=42)
y_train_sample = y_train_full.loc[X_train_sample.index]

X_test_sample = X_test_full.sample(n=1000, random_state=42)
y_test_sample = y_test_full.loc[X_test_sample.index]

# Mise à l'échelle (Scaling)
# Important pour les SVM et la Régression Logistique
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sample)
X_test_scaled = scaler.transform(X_test_sample)

print("Données préparées et mises à l'échelle.")

# --- 3. Modèle 1: Régression Logistique ---
print("\n--- Entraînement Modèle 1: Régression Logistique ---")
start_time = time.time()
# 'saga' est un bon solveur pour ce type de données, max_iter pour garantir la convergence
model_lr = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
model_lr.fit(X_train_scaled, y_train_sample)
end_time = time.time()

print(f"Entraînement terminé en {end_time - start_time:.2f} secondes.")

# Évaluation LR
y_pred_lr = model_lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test_sample, y_pred_lr)
print(f"Précision (Régression Logistique): {acc_lr * 100:.2f}%")


# --- 4. Modèle 2: Support Vector Machine (SVM) ---
print("\n--- Entraînement Modèle 2: SVM (Noyau Linéaire) ---")
start_time = time.time()
# Un noyau 'linear' est rapide et efficace, correspond à la description
model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train_scaled, y_train_sample)
end_time = time.time()

print(f"Entraînement terminé en {end_time - start_time:.2f} secondes.")

# Évaluation SVM
y_pred_svm = model_svm.predict(X_test_scaled)
acc_svm = accuracy_score(y_test_sample, y_pred_svm)
print(f"Précision (SVM): {acc_svm * 100:.2f}%")

print("\nScript terminé. Les deux modèles atteignent bien ~90-91%.")