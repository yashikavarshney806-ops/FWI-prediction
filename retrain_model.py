import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv('notebooks/Algerian_forest_fires_dataset.csv')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Clean data
df.dropna(how='all', inplace=True)
df = df.dropna().reset_index(drop=True)

# Drop unnecessary columns
df_copy = df.drop(['day', 'month', 'year'], axis=1)

# Convert Classes to binary
df_copy['Classes'] = np.where(df_copy['Classes'].str.contains('not fire'), 0, 1)

# Create features and target - EXCLUDING Classes
X = df_copy.drop(['FWI', 'Classes'], axis=1)
y = df_copy['FWI']

# Convert X to numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop any rows with NaN values created by conversion
valid_idx = ~(X.isna().any(axis=1) | y.isna())
X = X[valid_idx]
y = y[valid_idx]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {X.columns.tolist()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge model
ridge = Ridge()
ridge.fit(X_train_scaled, y_train)

# Evaluate
y_pred = ridge.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# Save models
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(ridge, open('models/ridge.pkl', 'wb'))

print("\nModels saved successfully!")
print(f"Expected input features: {X.shape[1]}")
