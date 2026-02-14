import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier # Cambiamos a Gradient Boosting (más preciso)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CARGA
print("--- CARGANDO Y PREPARANDO ---")
df = pd.read_csv("Dataset ADB-S.csv")
df = df.sort_values(by=['icao24', 'time'])

# 2. FEATURE ENGINEERING (Añadimos RSS y Rolling)
# Delta Tiempo y Física
df['dt'] = df.groupby('icao24')['time'].diff().fillna(10)
df['d_vel'] = df.groupby('icao24')['velocity'].diff().fillna(0)
df['acceleration'] = df['d_vel'] / df['dt'].replace(0, 0.1)

# Rumbo y Giro
df['d_heading'] = df.groupby('icao24')['heading'].diff().fillna(0)
df['d_heading'] = df['d_heading'].apply(lambda x: x if abs(x) <= 180 else 360 - abs(x))
df['turn_rate'] = df['d_heading'] / df['dt'].replace(0, 0.1)

# --- LA CLAVE PARA EL GHOSTING ---
# Calculamos la desviación estándar (varianza) en ventanas de 3 mensajes
# Un avión real "tiembla" en sus datos. Un fantasma digital suele ser "plano".
cols_std = ['velocity', 'heading', 'baroaltitude', 'rss'] # AÑADIMOS RSS
for col in cols_std:
    df[f'{col}_std'] = df.groupby('icao24')[col].rolling(3).std().reset_index(0, drop=True).fillna(0)

# 3. GESTIÓN DE NULOS
features = ['velocity', 'heading', 'vertrate', 'baroaltitude', 'rss', 'doppler', 
            'acceleration', 'turn_rate', 'dt', 
            'velocity_std', 'heading_std', 'baroaltitude_std', 'rss_std']

# Imputar Label 2 con media
media_label2 = df[df['label'] == 2][features].mean()
df.loc[df['label'] == 2, features] = df.loc[df['label'] == 2, features].fillna(media_label2)
df_clean = df.dropna(subset=features).copy()

# 4. SPLIT
aviones = df_clean['icao24'].unique()
np.random.shuffle(aviones)
split = int(len(aviones) * 0.7)
train_ids = aviones[:split]
test_ids = aviones[split:]

X_train = df_clean[df_clean['icao24'].isin(train_ids)][features]
y_train = df_clean[df_clean['icao24'].isin(train_ids)]['label']
X_test = df_clean[df_clean['icao24'].isin(test_ids)][features]
y_test = df_clean[df_clean['icao24'].isin(test_ids)]['label']

# 5. OVERSAMPLING MANUAL (La Trampa Maestra)
# Vamos a duplicar los casos de Ghosting (1) en el set de ENTRENAMIENTO
print("--- APLICANDO OVERSAMPLING A GHOSTING ---")
print(f"Original Train size: {len(X_train)}")

# Sacamos los índices de los ataques Ghost en el train
ghost_indices = y_train[y_train == 1].index

# Los duplicamos 2 veces más
X_ghost = X_train.loc[ghost_indices]
y_ghost = y_train.loc[ghost_indices]

# Los pegamos al set de entrenamiento (ahora la IA verá muchos más fantasmas)
X_train_balanced = pd.concat([X_train, X_ghost, X_ghost])
y_train_balanced = pd.concat([y_train, y_ghost, y_ghost])

print(f"New Train size (con clones de fantasmas): {len(X_train_balanced)}")

# 6. ENTRENAMIENTO (Gradient Boosting)
# Gradient Boosting suele detectar mejor patrones sutiles que Random Forest
print("--- ENTRENANDO GRADIENT BOOSTING ---")
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# 7. RESULTADOS
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("🎯 RESULTADOS FINALES (OVERSAMPLING + GRADIENT BOOSTING)")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Ghost(1)', 'Drift(2)', 'Flood(3)']))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges')
plt.title('Matriz Final')
plt.show()