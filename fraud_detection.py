import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Data prepration 

import pandas as pd
data = pd.read_csv('./creditcard.csv')
data.head()

# Exploratory data analysis

# Check for missing values
print(f"NaN values {data.isnull().sum()}")
# Drop rows with NaN values
data = data.dropna()
print(f"NaN values {data.isnull().sum()}")
# Check for class imbalance
fraud = data[data['Class'] == 1]
genuine = data[data['Class'] == 0]
print(f"Fraudulent transactions: {len(fraud)}, Genuine transactions: {len(genuine)}")

sns.countplot(data['Class'])
plt.show()

# Feature scaling
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# Split data into features and target variable
X = data.drop(['Class', 'Time', 'Amount'], axis=1) 
y = data['Class']

# Model Training

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")

# Store model
import joblib
joblib.dump(model, 'fraud_detection_model.pkl')

