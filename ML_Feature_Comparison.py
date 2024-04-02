import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Load the datasets
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# Separate features and targets
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on training set only
X_test = scaler.transform(X_test)  # Apply transform to both the training set and the test set

# Train Logistic Regression Model
logisticRegr = LogisticRegression(solver='lbfgs', max_iter=1000)  # Adjust max_iter as needed
logisticRegr.fit(X_train, y_train)
y_pred_lr = logisticRegr.predict(X_test)

# Train Random Forest Model
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

# Enhanced Confusion Matrix Plot
def plot_enhanced_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, ax=ax, square=True, linewidths=.5)
    ax.set(xlabel='Predicted label', ylabel='True label', title=title)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

# Plotting
plot_enhanced_confusion_matrix(y_test, y_pred_lr, classes=np.array(range(10)), title='Confusion matrix for Logistic Regression')
plot_enhanced_confusion_matrix(y_test, y_pred_rf, classes=np.array(range(10)), title='Confusion matrix for Random Forest')

plt.show()

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report for Logistic Regression:\n", classification_report(y_test, y_pred_lr))
print("\nClassification Report for Random Forest:\n", classification_report(y_test, y_pred_rf))
