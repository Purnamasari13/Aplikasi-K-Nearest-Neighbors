import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Konfigurasi visualisasi
sns.set_theme(style="whitegrid", palette="viridis")

# Lokasi direktori
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, 'static')
os.makedirs(static_dir, exist_ok=True)

# Memuat data dan urutkan berdasarkan kolom Id
csv_path = os.path.join(current_dir, 'iris.csv')
try:
    df = pd.read_csv(csv_path)
    print("Dataset berhasil dimuat.")
    df = df.sort_values(by='Id').reset_index(drop=True)  # Mengurutkan berdasarkan ID
except FileNotFoundError:
    print(f"Gagal menemukan file: {csv_path}")
    exit()

# Bersihkan kolom
df.columns = [col.strip().lower().replace(' ', '') for col in df.columns]

# Pisahkan fitur dan label
feature_cols = ['sepallengthcm', 'sepalwidthcm', 'petallengthcm', 'petalwidthcm']
X = df[feature_cols]
y = df['species']

# Visualisasi data
plt.figure(figsize=(10, 7))
sns.scatterplot(x='sepallengthcm', y='sepalwidthcm', hue='species', data=df, s=100)
plt.title('Scatter Plot Panjang vs Lebar Sepal')
plt.savefig(os.path.join(static_dir, 'scatter_sepal.png'))
plt.show()

sns.pairplot(df, hue='species')
plt.savefig(os.path.join(static_dir, 'pairplot.png'))
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Korelasi antar Fitur')
plt.savefig(os.path.join(static_dir, 'heatmap_korelasi.png'))
plt.show()

# Boxplot
plt.figure(figsize=(12, 8))
for i, col in enumerate(feature_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=col, data=df)
    plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'boxplot_fitur.png'))
plt.show()

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Cross-validation untuk cari k terbaik
k_values = range(1, 21)
cv_accuracies = []
best_k = 1
best_acc = 0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=kf)
    mean_acc = np.mean(scores)
    cv_accuracies.append(mean_acc)
    if mean_acc > best_acc:
        best_acc = mean_acc
        best_k = k

print(f"Nilai k optimal: {best_k} dengan akurasi: {best_acc:.4f}")

# Plot akurasi vs k
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_accuracies, marker='o')
plt.axvline(best_k, color='red', linestyle='--', label=f'k terbaik = {best_k}')
plt.title('Elbow Method untuk Menentukan k')
plt.xlabel('Jumlah Tetangga (k)')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'elbow_method.png'))
plt.show()

# Latih model akhir
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)

# Evaluasi
y_pred = final_model.predict(X_test_scaled)
print(f"Akurasi Test: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=final_model.classes_, yticklabels=final_model.classes_)
plt.title('Confusion Matrix')
plt.savefig(os.path.join(static_dir, 'confusion_matrix.png'))
plt.show()

# Visualisasi hasil prediksi
results_df = X_test.copy()
results_df['Actual'] = y_test
results_df['Predicted'] = y_pred
results_df['Correct'] = results_df['Actual'] == results_df['Predicted']

plt.figure(figsize=(10, 7))
sns.scatterplot(x='sepallengthcm', y='sepalwidthcm',
                hue='Actual', style='Correct',
                data=results_df, s=100, alpha=0.8,
                palette={'Iris-setosa': 'blue', 'Iris-versicolor': 'orange', 'Iris-virginica': 'green'},
                markers={True: 'o', False: 'X'})
plt.title('Hasil Prediksi Sepal')
plt.tight_layout()
plt.savefig(os.path.join(static_dir, 'hasil_prediksi.png'))
plt.show()

# Simpan model dan scaler
joblib.dump(final_model, os.path.join(current_dir, 'model.pkl'))
joblib.dump(scaler, os.path.join(current_dir, 'scaler.pkl'))
print("Model dan scaler berhasil disimpan.")
