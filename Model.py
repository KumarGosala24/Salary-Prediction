# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# import joblib

# # --------------------------
# # Load Dataset
# # --------------------------
# data = pd.read_csv("Salary Data.csv")  # make sure file name is correct

# # Basic info
# print(data.head(10))
# print(data.shape)
# print(data.tail(3))

# # Checking missing values
# print(data.isna().sum())

# # Replace '?' with 'Others' for workclass and occupation
# data['workclass'].replace({'?': 'Others'}, inplace=True)
# data['occupation'].replace({'?': 'Others'}, inplace=True)

# # Filter unwanted workclass categories
# data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]

# # Age
# plt.boxplot(data['age']); plt.title("Age"); plt.show()
# data = data[(data['age'] >= 17) & (data['age'] <= 75)]
# plt.boxplot(data['age']); plt.title("Cleaned Age"); plt.show()

# # Capital Gain
# plt.boxplot(data['capital-gain']); plt.title("Capital Gain"); plt.show()

# # Educational Num
# plt.boxplot(data['educational-num']); plt.title("Educational Number"); plt.show()
# data = data[(data['educational-num'] >= 5) & (data['educational-num'] <= 16)]
# plt.boxplot(data['educational-num']); plt.title("Cleaned Educational Num"); plt.show()

# # Hours Per Week
# plt.boxplot(data['hours-per-week']); plt.title("Hours per Week"); plt.show()

# # Drop redundant column
# data.drop(columns=['education'], inplace=True)

# # --------------------------
# # Encoding categorical variables
# # --------------------------
# encoders = {}
# categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

# for col in categorical_cols:
#     le = LabelEncoder()
#     data[col] = le.fit_transform(data[col])
#     encoders[col] = le  # save encoder for later use in Streamlit

# # --------------------------
# # Split data
# # --------------------------
# X = data.drop(columns=['income'])
# y = data['income']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --------------------------
# # Models
# # --------------------------
# models = {
#     "LogisticRegression": LogisticRegression(),
#     "RandomForest": RandomForestClassifier(),
#     "KNN": KNeighborsClassifier(),
#     "SVM": SVC(),
#     "GradientBoosting": GradientBoostingClassifier()
# }

# results = {}

# # --------------------------
# # Train & Evaluate
# # --------------------------
# for name, model in models.items():
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('model', model)
#     ])
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     results[name] = acc
#     print(f"{name} Accuracy: {acc:.4f}")
#     print(classification_report(y_test, y_pred))

# # --------------------------
# # Plot accuracy comparison
# # --------------------------
# plt.bar(results.keys(), results.values(), color='skyblue')
# plt.ylabel('Accuracy Score')
# plt.title('Model Comparison')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()

# # --------------------------
# # Select best model & save
# # --------------------------
# best_model_name = max(results, key=results.get)
# best_model = models[best_model_name]
# best_model.fit(X_train, y_train)  # train again on full train set

# print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# joblib.dump(best_model, "best_model.pkl")
# joblib.dump(encoders, "encoders.pkl")  # save encoders for later use
# print("✅ Saved best model as best_model.pkl")
# print("✅ Saved encoders as encoders.pkl")



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
import joblib

# --------------------------
# Load Dataset
# --------------------------
data = pd.read_csv("Salary Data.csv")

# Basic info
print(data.head())
print("\nShape:", data.shape)

# --------------------------
# Handle Missing Values
# --------------------------
# Fill numeric columns with median, categorical with mode
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].median())

print("\nMissing values after filling:\n", data.isna().sum())

# --------------------------
# Encoding categorical variables
# --------------------------
encoders = {}
categorical_cols = ['Gender', 'Education Level', 'Job Title']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# --------------------------
# Split data
# --------------------------
X = data.drop(columns=['Salary'])
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Regression Models
# --------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SVR": SVR(),
    "GradientBoostingRegressor": GradientBoostingRegressor()
}

results = {}

# --------------------------
# Train & Evaluate
# --------------------------
for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = r2
    print(f"{name} -> R2 Score: {r2:.4f}, MSE: {mse:.2f}")

# --------------------------
# Plot R2 score comparison
# --------------------------
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel('R2 Score')
plt.title('Regression Model Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# --------------------------
# Save best model
# --------------------------
best_model_name = max(results, key=results.get)
best_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', models[best_model_name])
])
best_model.fit(X_train, y_train)

print(f"\n✅ Best model: {best_model_name} with R2 Score {results[best_model_name]:.4f}")

joblib.dump(best_model, "best_salary_model.pkl")
joblib.dump(encoders, "salary_encoders.pkl")
print("✅ Saved best model as best_salary_model.pkl")
print("✅ Saved encoders as salary_encoders.pkl")
