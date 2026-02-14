import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. CARGA
df = pd.read_csv("Dataset ADB-S.csv")

# 2. LIMPIEZA RÁPIDA (Basada en tu Fase 1)
# Eliminamos lo que acordamos que era ruido o identificadores directos
cols_to_drop = ['callsign', 'spi', 'icao24', 'time'] 
# Nota: Quitamos time e icao24 del entrenamiento para que aprenda FÍSICA, no ID o Tiempo.
X = df.drop(columns=cols_to_drop + ['label'])
y = df['label']

# 3. SPLIT POR AVIONES (Tu estrategia Maestra)
# Para este test rápido, lo haremos simple pero respetando la separación
aviones = df['icao24'].unique()
train_ids, test_ids = train_test_split(aviones, test_size=0.3, random_state=42)

X_train = df[df['icao24'].isin(train_ids)].drop(columns=cols_to_drop + ['label'])
y_train = df[df['icao24'].isin(train_ids)]['label']
X_test = df[df['icao24'].isin(test_ids)].drop(columns=cols_to_drop + ['label'])
y_test = df[df['icao24'].isin(test_ids)]['label']

# Manejo simple de nulos para que el script no falle
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

print(f"Entrenando con {len(X_train)} muestras y probando con {len(X_test)}...")

# 4. MODELO
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. PREDICCIÓN Y EVALUACIÓN
y_pred = model.predict(X_test)

print("\n" + "="*30)
print("RESULTADOS DEL ")
print("="*30)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score (Macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))

# 6. MATRIZ DE CONFUSIÓN (Para ver dónde se confunde)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.show()