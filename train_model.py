
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("ckd_dataset.csv")
df.columns = df.columns.str.strip()

target = "classification"

if target not in df.columns:
    raise ValueError(
        f"Target '{target}' not found. Columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=[target])
X = df.drop(columns=[target])
y = df[target].astype(str).str.strip()

# Convert categorical columns into numeric columns
X = pd.get_dummies(X, dummy_na=True)
X = X.replace([float("inf"), float("-inf")], float("nan"))

if y.nunique() < 2:
    raise ValueError("Training requires at least two target classes.")

stratify = y if y.value_counts().min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
)

model.fit(X_train, y_train)

joblib.dump(
    {"model": model, "columns": X.columns.tolist()},
    "ckd_model.pkl"
)

print("Model trained successfully.")
print("Test accuracy:", model.score(X_test, y_test))
