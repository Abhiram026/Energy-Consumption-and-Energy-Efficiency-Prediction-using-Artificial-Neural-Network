from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def admin():
    return render_template("login.html")


USERNAME = 'admin'
PASSWORD = 'admin'
uploaded_data = None  # Global variable to hold data

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    if username == USERNAME and password == PASSWORD:
        return redirect(url_for('home'))
    else:
        return render_template('login.html', error='Invalid credentials')

@app.route('/home', methods=['GET', 'POST'])
def home():
    global uploaded_data

    top_rows = None

    if request.method == 'POST' and 'dataset' in request.files:
        file = request.files['dataset']
        if file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            uploaded_data = df.head().to_html(classes='data', header="true")
            top_rows = uploaded_data
        else:
            top_rows = "<p style='color:red'>Invalid file type. Please upload a CSV file.</p>"

    return render_template('home.html', top_rows=top_rows)
import pandas as pd

# Global variable to store the latest uploaded filename
uploaded_filename = None

@app.route('/upload', methods=['POST'])
def upload_dataset():
    global uploaded_filename
    file = request.files['dataset']
    if file and file.filename.endswith('.csv'):
        uploaded_filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename))
        message = "Dataset loaded successfully!"
    else:
        message = "Invalid file format. Please upload a .csv file."
    return render_template('home.html', message=message)

@app.route('/view-dataset', methods=['POST'])
def view_dataset():
    global uploaded_filename
    if not uploaded_filename:
        return render_template('home.html', message="No dataset uploaded yet.")
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
    df = pd.read_csv(filepath)

    head = df.head().to_html(classes='data', header="true")
    missing = df.isnull().sum().to_frame(name='Missing Values').to_html(classes='data', header=True)
    shape_info = f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."

    return render_template('view_dataset.html', head=head, missing=missing, shape_info=shape_info)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    global uploaded_filename, X_train_glob, X_test_glob, y1_train_glob, y1_test_glob, y2_train_glob, y2_test_glob,scaler

    if not uploaded_filename:
        return render_template('home.html', message="No dataset uploaded yet.")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
    df = pd.read_csv(filepath)

    X = df.drop(['Heating_Load', 'Cooling_Load'], axis=1)
    Y1 = df[['Heating_Load']]
    Y2 = df[['Cooling_Load']]

    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, Y1, Y2, test_size=0.33, random_state=20)

    MinMax = MinMaxScaler(feature_range=(0, 1))
    X_train = MinMax.fit_transform(X_train)
    X_test = MinMax.transform(X_test)

    scaler = MinMax  # Save scaler for prediction

    # Save to global
    X_train_glob = X_train
    X_test_glob = X_test
    y1_train_glob = y1_train
    y1_test_glob = y1_test
    y2_train_glob = y2_train
    y2_test_glob = y2_test

    shapes = {
        'X_train': X_train.shape,
        'X_test': X_test.shape,
        'y1_train': y1_train.shape,
        'y1_test': y1_test.shape,
        'y2_train': y2_train.shape,
        'y2_test': y2_test.shape
    }

    return render_template('home.html', shapes=shapes)


@app.route('/train', methods=['POST'])
def train_model():
    global X_train_glob, X_test_glob, y1_train_glob, y1_test_glob, y2_train_glob, y2_test_glob
    global model1, model2

    if X_train_glob is None:
        return render_template('home.html', message="Please preprocess data first.")

    # Train for Heating Load
    model1 = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=1)
    model1.fit(X_train_glob, y1_train_glob.values.ravel())
    y1_pred = model1.predict(X_test_glob)

    # Train for Cooling Load
    model2 = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=2)
    model2.fit(X_train_glob, y2_train_glob.values.ravel())

    mse = mean_squared_error(y1_test_glob, y1_pred)
    r2 = r2_score(y1_test_glob, y1_pred)

    y1_pred_int = np.round(y1_pred)
    y1_test_int = np.round(y1_test_glob.values.ravel())
    accuracy = accuracy_score(y1_test_int, y1_pred_int)

    results = {
        'mse': round(mse, 4),
        'r2': round(r2, 4),
        'accuracy': round(accuracy, 4)
    }

    return render_template('home.html', results=results)




@app.route('/predict', methods=['POST'])
def predict():
    global model1, model2, scaler

    if not model1 or not model2 or not scaler:
        return render_template('home.html', message="Model is not trained yet.")

    file = request.files['predict_file']
    if file and file.filename.endswith('.csv'):
        input_df = pd.read_csv(file)

        # Scale input data
        input_scaled = scaler.transform(input_df)

        # Predict using trained models
        y1_pred = model1.predict(input_scaled)
        y2_pred = model2.predict(input_scaled)

        # Combine predictions
        predictions = list(zip(np.round(y1_pred, 2), np.round(y2_pred, 2)))

        return render_template('home.html', predictions=predictions)
    
    return render_template('home.html', message="Invalid file format. Please upload a CSV file.")


@app.route('/view_graphs', methods=['POST'])
def view_graphs():
    global uploaded_filename
    if not uploaded_filename:
        return render_template('home.html', message="Please upload a dataset first.")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
    df = pd.read_csv(filepath)

    num_list = list(df.columns)
    fig = plt.figure(figsize=(10, 30))

    for i in range(len(num_list)):
        plt.subplot(15, 2, i + 1)
        plt.title(num_list[i])
        plt.hist(df[num_list[i]], color='blue', alpha=0.5)

    plt.tight_layout()

    # Convert to image and embed
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('graphs.html', graph_url=graph_url)


@app.route('/logout', methods=['POST'])
def logout():
    return redirect(url_for('admin'))


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
