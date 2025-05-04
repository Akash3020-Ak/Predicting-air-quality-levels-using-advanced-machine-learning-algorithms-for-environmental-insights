import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('air_quality.csv').dropna()
X = df[['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']]
y = df['C6H6(GT)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

sample = [[2.0, 150.0, 120.0, 20.0, 40.0, 0.8]]
print(model.predict(scaler.transform(sample))[0])
