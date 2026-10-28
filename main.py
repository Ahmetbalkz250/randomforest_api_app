import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import accuracy_score, f1_score

sutun_isimleri = ['buying_price', 'maintenance_cost', 'number_of_doors', 'number_of_persons', 'lug_boot', 'safety', 'quality']

df = pd.read_csv("car_evaluation.csv", header=None, names=sutun_isimleri)

fiyat_sirasi = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
kapi_sirasi = {'2': 0, '3': 1, '4': 2, '5more': 3}
kisi_sirasi = {'2': 0, '4': 1, 'more': 2}
bagaj_sirasi = {'small': 0, 'med': 1, 'big': 2}
guvenlik_sirasi = {'low': 0, 'med': 1, 'high': 2}
kalite_sirasi = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}


df['buying_price'] = df['buying_price'].map(fiyat_sirasi)
df['maintenance_cost'] = df['maintenance_cost'].map(fiyat_sirasi)
df['number_of_doors'] = df['number_of_doors'].map(kapi_sirasi)
df['number_of_persons'] = df['number_of_persons'].map(kisi_sirasi)
df['lug_boot'] = df['lug_boot'].map(bagaj_sirasi)
df['safety'] = df['safety'].map(guvenlik_sirasi)
df['quality'] = df['quality'].map(kalite_sirasi)


X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = rf(n_estimators=100, max_depth=10, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

f1 = f1_score(y_test, y_pred, average='weighted')
acc = accuracy_score(y_test, y_pred)
print(f1)
print(acc)
import joblib
joblib.dump(model, "rf_car_quality.pkl")