import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Verilerin yüklenmesi
data_dict = pickle.load(open("data.pickle", "rb"))
data = data_dict["data"]

# Tüm öğelerin aynı uzunluğa getirilmesi
max_length = max(len(item) for item in data)
for i in range(len(data)):
    while len(data[i]) < max_length:
        data[i].append([0.0, 0.0])
    data[i] = data[i][:max_length]

# Veri yapısının kontrol edilmesi
for i in range(len(data)):
    print(f"Veri {i} Uzunluğu: {len(data[i])}")

# Veri setinin düzeltilmiş bir şekilde NumPy dizisine dönüştürülmesi
data_np = np.array([np.array(item).reshape(-1) for item in data])

# Etiketlerin hazırlanması
labels = data_dict["labels"]

# Veri setinin eğitim ve test olarak ayrılması
x_train, x_test, y_train, y_test = train_test_split(data_np, labels, test_size=0.2, shuffle=True, stratify=labels)

# Model oluşturma ve eğitme (21 özellik kullanarak)
model = RandomForestClassifier(n_estimators=100, max_features=21)
model.fit(x_train, y_train)

# Tahmin yapma
y_pred = model.predict(x_test)

# Doğruluk değerinin hesaplanması
score = accuracy_score(y_pred, y_test)
print("Başarı:", score * 100)

# Modelin kaydedilmesi
with open("model.p", "wb") as f:
    pickle.dump({"model": model}, f)













