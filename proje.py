import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv("C:\\Users\\fatih selman\\Downloads\\diabetes.csv")

X = data.iloc[:, :-1]  # Tüm sütunlar (özellikler) hariç son sütun
y = data.iloc[:, -1]   # Son sütun (hedef değişken)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Kullanıcıdan veri almak için input fonksiyonunu kullan
print("\nLütfen aşağıdaki özellikler için değerleri girin:")

pregnancies = float(input("Gebelik sayısı: "))
glucose = float(input("Glukoz: "))
blood_pressure = float(input("Kan Basıncı: "))
skin_thickness = float(input("Cilt Kalınlığı: "))
insulin = float(input("İnsülin: "))
bmi = float(input("BMI: "))
diabetes_pedigree = float(input("Genetik yatkınlık: "))
age = float(input("Yaş: "))

# Kullanıcıdan alınan verileri bir listeye koy
user_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]

# Veriyi ölçeklendir
user_data = scaler.transform(user_data)

# Tahmin yap
prediction = model.predict(user_data)

if prediction[0] == 1:
    print("\nTahmin: Diyabetli olabilirsiniz.")
else:
    print("\nTahmin: Diyabetli değilsiniz.")
