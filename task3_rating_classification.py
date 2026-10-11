
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Step 1: Load dataset
df = pd.read_csv("DATASET.csv")

print("Dataset loaded:", df.shape)

# Step 2: Prepare the target
# Combine very rare rating categories into Other
rating_counts = df["rating"].value_counts()
df["rating_target"] = df["rating"].where(
    df["rating"].map(rating_counts) >= 30,
    "Other"
)

print("\nRating categories:")
print(df["rating_target"].value_counts())

# Step 3: Select input features
numeric_features = ["release_year"]

categorical_features = [
    "type",
    "director",
    "country",
    "listed_in"
]

X = df[numeric_features + categorical_features]
y = df["rating_target"]

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Step 5: Prepare numerical and categorical data
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=2
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# Step 6: Define classification models
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=16,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_accuracy = -1
best_model_name = None
best_predictions = None

# Step 7: Train and evaluate
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4)
    })

    print(f"{name} accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_predictions = predictions

# Step 8: Compare models
results_df = pd.DataFrame(results)

print("\nModel comparison:")
print(results_df.to_string(index=False))

print("\nBest model:", best_model_name)
print("\nClassification report:")
print(classification_report(
    y_test,
    best_predictions,
    zero_division=0
))

# Step 9: Save results
results_df.to_csv(
    "outputs/task3_model_comparison.csv",
    index=False
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    xticks_rotation=45,
    colorbar=False
)

plt.title(f"Rating Classification - {best_model_name}")
plt.tight_layout()
plt.savefig(
    "outputs/task3_confusion_matrix.png",
    dpi=150
)
plt.close()

print("\nTask 3 completed successfully!")
