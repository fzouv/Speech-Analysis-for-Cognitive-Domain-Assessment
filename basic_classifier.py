import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


file_path = '/Users/fotianazouvani/Downloads/SLP/Diss/linguistic_and_acoustic.csv'
data = pd.read_csv(file_path)
data.sample(5)

#preprocessing data
#converting labels to binary (0 for HC and 1 for MCI)
data['Group'] = data['Group'].map({'Control': 0, 'MCI': 1})


X = data.drop(['File', 'Language', 'Corpus', 'Code', 'Age', 'Sex', 'Group', 'Role', 'MoCA scores'], axis=1)
y = data['Group']

#training and test sets split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#standardising features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#building and training logistic regression model
model = LogisticRegression(random_state=42, max_iter=2000)  #increasing iteration steps
model.fit(X_train, y_train)

#evaluating the model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Baseline model: Predicting the most frequent class
most_frequent_class = y_train.mode()[0]
y_pred_baseline = [most_frequent_class] * len(y_test)

print("\nBaseline Model Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_baseline))
print("Classification Report:\n", classification_report(y_test, y_pred_baseline, zero_division=1))

# save predictions to cvs
test_indices = y_test.index
predictions = pd.DataFrame({
    'File': data['File'].iloc[test_indices],
    'Actual': y_test.values,
    'Predicted': y_pred
})
predictions.to_csv('predictions_linguistic_and_acoustic.csv', index=False)
