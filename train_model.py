import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Drop non-predictive columns
df = df.drop("id", axis=1)

# Fill missing BMI with median
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

X = df.drop("stroke", axis=1)
y = df["stroke"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Save
joblib.dump(X.columns.tolist(), "columns.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "stroke_model.pkl")
