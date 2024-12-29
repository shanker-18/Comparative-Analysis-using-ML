import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
from flask import Flask, request, jsonify

# Step 1: Dataset Preparation
def prepare_dataset():
    # Example dataset for regression, classification, and clustering
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'label_regression': np.random.rand(100) * 100,
        'label_classification': np.random.choice([0, 1], 100)
    }
    return pd.DataFrame(data)

# Step 2: Regression Analysis
def regression_analysis(data):
    X = data[['feature1', 'feature2']]
    y = data['label_regression']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Step 3: Classification Analysis
def classification_analysis(data):
    X = data[['feature1', 'feature2']]
    y = data['label_classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Step 4: Clustering Analysis
def clustering_analysis(data):
    X = data[['feature1', 'feature2']]
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X)
    silhouette_avg = silhouette_score(X, model.labels_)
    return silhouette_avg

# Step 5: Flask API Setup
app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    # Prepare dataset
    data = prepare_dataset()
    
    # Perform analyses
    mse = regression_analysis(data)
    accuracy = classification_analysis(data)
    silhouette_avg = clustering_analysis(data)

    # Return results
    response = {
        "regression_mse": mse,
        "classification_accuracy": accuracy,
        "clustering_silhouette_score": silhouette_avg
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

# Step-by-Step in Cells

# Cell 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score

# Cell 2: Dataset preparation
def prepare_dataset():
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'label_regression': np.random.rand(100) * 100,
        'label_classification': np.random.choice([0, 1], 100)
    }
    return pd.DataFrame(data)

data = prepare_dataset()

# Cell 3: Regression analysis
def regression_analysis(data):
    X = data[['feature1', 'feature2']]
    y = data['label_regression']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

mse = regression_analysis(data)
print("Regression MSE:", mse)

# Cell 4: Classification analysis
def classification_analysis(data):
    X = data[['feature1', 'feature2']]
    y = data['label_classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

accuracy = classification_analysis(data)
print("Classification Accuracy:", accuracy)

# Cell 5: Clustering analysis
def clustering_analysis(data):
    X = data[['feature1', 'feature2']]
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X)
    silhouette_avg = silhouette_score(X, model.labels_)
    return silhouette_avg

silhouette_avg = clustering_analysis(data)
print("Clustering Silhouette Score:", silhouette_avg)

# Cell 6: Flask API setup
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = prepare_dataset()
    mse = regression_analysis(data)
    accuracy = classification_analysis(data)
    silhouette_avg = clustering_analysis(data)
    response = {
        "regression_mse": mse,
        "classification_accuracy": accuracy,
        "clustering_silhouette_score": silhouette_avg
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

# Cell 7: Testing with Postman
# Use Postman to send POST request to `http://127.0.0.1:5000/analyze`
