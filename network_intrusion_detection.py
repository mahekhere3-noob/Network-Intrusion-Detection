import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
# Expected columns: Duration, Src_Bytes, Dst_Bytes, Count, Failed_Logins, Attack (0 = Normal, 1 = Attack)
data = pd.read_csv("network_intrusion_data.csv")

# Features and Target
X = data[["Duration", "Src_Bytes", "Dst_Bytes", "Count", "Failed_Logins"]]
y = data["Attack"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test Accuracy
prediction = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, prediction))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))
print("\nClassification Report:")
print(classification_report(y_test, prediction, target_names=["Normal", "Attack"]))

# User Prediction
duration = float(input("Enter Connection Duration (seconds): "))
src_bytes = float(input("Enter Source Bytes: "))
dst_bytes = float(input("Enter Destination Bytes: "))
count = float(input("Enter Connection Count (past 2 sec, same host): "))
failed_logins = float(input("Enter Number of Failed Logins: "))

new_data = [[duration, src_bytes, dst_bytes, count, failed_logins]]

result = model.predict(new_data)

if result[0] == 1:
    print("\nPrediction: Network Traffic looks like an ATTACK 🚨")
else:
    print("\nPrediction: Network Traffic looks NORMAL ✅")
