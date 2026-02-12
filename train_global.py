import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import os

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

print("Fetching California Housing data...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# --- Feature Engineering (Synthetic) ---
# Goal: Augment Lat/Lon with Area, Garage, Bedrooms, Age
# We treat original target (MedHouseVal) as "Location Value"
# We add "Structure Value" based on synthetic features.

print("Generating synthetic structural features...")
np.random.seed(42)
n_samples = len(df)

# features: Latitude, Longitude, TotalArea, GarageCars, Bedrooms, HouseAge
df['TotalArea'] = np.random.normal(1800, 600, n_samples).astype(int)
df['TotalArea'] = df['TotalArea'].clip(500, 5000)

df['GarageCars'] = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.1, 0.3, 0.5, 0.1])
df['Bedrooms'] = np.random.randint(1, 6, size=n_samples)

# Rename HouseAge
df['HouseAge'] = df['HouseAge']

# --- Target Adjustment ---
# Logic: Price = LocationBase + StructureValue
# MedHouseVal is in $100k units.
# Let's assume MedHouseVal captures the "Location Premium".
# We add structure value to it.

# Coefficients (in $100k units):
# Area: $150/sqft = 0.0015
# Garage: $15k = 0.15
# Bedroom: $10k = 0.10
# Age: -$2k/year = -0.02

# Base Location Value (from dataset, clipped to reasonable range 0.5-5.0)
location_component = df['MedHouseVal']

structure_component = (
    (df['TotalArea'] * 0.0015) + 
    (df['GarageCars'] * 0.15) + 
    (df['Bedrooms'] * 0.10) - 
    (df['HouseAge'] * 0.02)
)

# New Target
df['FinalPrice'] = location_component + structure_component
# Ensure no negative prices
df['FinalPrice'] = df['FinalPrice'].clip(lower=0.5)

# Select Final Features
feature_cols = ['Latitude', 'Longitude', 'TotalArea', 'GarageCars', 'Bedrooms', 'HouseAge']
X = df[feature_cols]
y = df['FinalPrice']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature_cols)
    ])

# Model Pipeline
# Using simpler XGBoost params for speed/stability
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1))
])

# Train
print(f"Training Hybrid Model on {len(df)} samples...")
model.fit(X, y)

# Save
print("Saving model to data/global_model.joblib...")
joblib.dump(model, 'data/global_model.joblib')
print("Done.")
