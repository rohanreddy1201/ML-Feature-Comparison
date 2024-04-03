# ML Feature Comparison

This project compares the performance of Logistic Regression and Random Forest models trained on the MNIST dataset. 
It includes data loading, preprocessing, model training, evaluation, and visualization.

# Dataset:
The MNIST dataset is a collection of 28x28 pixel grayscale images of handwritten digits (0 to 9).
It consists of two CSV files: mnist_train.csv (60,000 images) and mnist_test.csv (10,000 images).

# Models:
1. Logistic Regression: A linear model commonly used for binary classification, extended to multi-class classification.
2. Random Forest: An ensemble learning method that constructs a multitude of decision trees during training.

# Evaluation Metrics:
1. Accuracy: Proportion of correctly classified instances.
2. Precision: Proportion of true positive predictions out of all positive predictions.
3. Recall: Proportion of true positive predictions out of all actual positive instances.
4. F1-score: Harmonic mean of precision and recall, providing a balanced measure.

# Visualization:
1. Confusion Matrix: A matrix showing the counts of true positive, false positive, true negative, and false negative predictions.
2. Enhanced Confusion Matrix: A heatmap visualization of the confusion matrix with precision, recall, and F1-score statistics.

# Dependencies:
- numpy: For numerical operations on arrays.
- pandas: For data manipulation and CSV file loading.
- matplotlib: For data visualization, including plotting confusion matrices.
- seaborn: For advanced heatmap visualization.
- scikit-learn: For machine learning algorithms and evaluation metrics.

# Usage:
1. Clone the repository.
2. Run the script `ML_Feature_Comparison.py` to train the models and evaluate their performance.
3. View the results in the console and visualizations in separate windows.

# Disclaimer:
This project is provided for educational purposes only. The code and information included in this repository are intended to demonstrate concepts and techniques in machine learning and data analysis. It is not intended for direct implementation in real-world applications without proper validation, testing, and adaptation to specific use cases.
