import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tqdm.notebook import tqdm  # Use tqdm for progress visualization

# 1. Load and Preprocess Data
df = pd.read_csv("diabetes_prediction_dataset.csv")  

df.drop_duplicates(inplace=True)

print("Missing values in each column:\n", df.isna().sum())

categorical_columns = ["gender", "smoking_history"]
le_dict = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Conv1D deep learning model reshaping the input to (samples, time_steps, channels)
X_dl = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# 2. Model Definitions
# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# SVM Model
svm_model = SVC(kernel='linear', probability=True, random_state=42)

# Conv1D Deep Learning Model
def create_conv1d_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Metrics Calculation Function
def calculate_metrics(cm):
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0
    TSS = TPR - FPR 
    denominator = ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
    HSS = 2 * (TP * TN - FN * FP) / denominator if denominator != 0 else 0
    return {"accuracy": accuracy, "TPR": TPR, "FPR": FPR, "FNR": FNR, "TSS": TSS, "HSS": HSS,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN}

# 4. KFold Cross-Validation Setup
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

rf_metrics_list = []
svm_metrics_list = []
conv1d_metrics_list = []

# 5. Training and Evaluation using KFold with tqdm
print("Training Random Forest Model:")
for i, (train_index, test_index) in enumerate(tqdm(kf.split(X), total=n_splits, desc="RF CV"), start=1):
    # Split the data based on current fold indices
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Measure training time
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Fold {i} RF training time: {train_time:.6f} seconds")
    
    # Prediction and metrics calculation
    y_pred = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    rf_metrics_list.append(calculate_metrics(cm))

print("\nTraining SVM Model:")
for i, (train_index, test_index) in enumerate(tqdm(kf.split(X), total=n_splits, desc="SVM CV"), start=1):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    start_time = time.time()
    svm_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Fold {i} SVM training time: {train_time:.6f} seconds")
    
    y_pred = svm_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    svm_metrics_list.append(calculate_metrics(cm))

print("\nTraining Conv1D Model:")
for i, (train_index, test_index) in enumerate(tqdm(kf.split(X), total=n_splits, desc="Conv1D CV"), start=1):
    X_train_dl, X_test_dl = X_dl[train_index], X_dl[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Create a new model instance for each fold to ensure fresh weights
    model = create_conv1d_model((X_train_dl.shape[1], 1))
    start_time = time.time()
    model.fit(X_train_dl, y_train, epochs=20, batch_size=16, verbose=0)
    train_time = time.time() - start_time
    print(f"Fold {i} Conv1D training time: {train_time:.6f} seconds")
    
    y_pred = (model.predict(X_test_dl) > 0.5).astype("int32").flatten()
    cm = confusion_matrix(y_test, y_pred)
    conv1d_metrics_list.append(calculate_metrics(cm))

# 6. Aggregate and Display Results in a Table
def aggregate_metrics(metrics_list):
    return pd.DataFrame(metrics_list).mean()

rf_results = aggregate_metrics(rf_metrics_list)
svm_results = aggregate_metrics(svm_metrics_list)
conv1d_results = aggregate_metrics(conv1d_metrics_list)

results_df = pd.DataFrame({
    "Random Forest": rf_results,
    "SVM": svm_results,
    "Conv1D": conv1d_results
}).transpose()

print("\nAverage Performance Metrics (10-Fold CV):")
print(results_df)
