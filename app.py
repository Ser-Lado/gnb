from collections import Counter
import io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from flask import Flask, redirect, render_template, request, session, url_for
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sqlite3
import bcrypt
from flask_recaptcha import ReCaptcha
from jinja2 import Markup


# Declare a Flask app
app = Flask(__name__)
app.secret_key = '73dD@kjfkd_sdS=s'
app.config['UPLOAD_FOLDER'] = 'batchCSV'

app.config['RECAPTCHA_PUBLIC_KEY'] = '6Ldd-50lAAAAAMFBkrIviLD-8MQR2vhP2L-ajMiH'
app.config['RECAPTCHA_PRIVATE_KEY'] = '6Ldd-50lAAAAADuzcMriCJDvYJ9Or8ayYB9dBRiu'
recaptcha = ReCaptcha(app)


# SQLite database connection for GRAPH
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()


# Functions for BAR Graph
def get_data_from_db():
    conn = sqlite3.connect('data.db', check_same_thread=False)
    df = pd.read_sql_query("SELECT * from datasets", conn)
    conn.close()
    return df

# function to preprocess data and train model
def preprocess_and_train_model(df):
    X = df[['GWA', 'preBoardLETscore']]
    y = df['outcome']
    model = GaussianNB()
    model.fit(X, y)
    return model

# function to generate bar chart
def generate_bar_chart(data):
    fig = go.Figure(
        data=[go.Bar(x=data.index, y=data.values)],
        layout=go.Layout(title='Bar Chart', xaxis=dict(title='Outcome'), yaxis=dict(title='Count'))
    )
    return fig.to_html(full_html=False)
# This code is for BAR Graph - END ************


def train_model():
    conn = sqlite3.connect('data.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT GWA, preBoardLETscore, outcome FROM datasets")
    data = cursor.fetchall()
    X = []
    y = []
    for row in data:
        X.append([float(row[0]), float(row[1])])
        y.append(row[2])
    gnb = GaussianNB()
    gnb.fit(X, y)
    conn.close()
    return gnb

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    else:
        return redirect(url_for('login'))
    

# Route for analyzing a single student's data
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if 'username' not in session:
        # If the user is not logged in, redirect them to the login page
        return redirect(url_for('login'))

    # Load data from SQLite database
    df = pd.read_sql_query("SELECT * FROM datasets", conn)

    # Split the data into features and target variable
    X = df[['GWA', 'preBoardLETscore']]
    y = df['outcome']

    # Train the model using Gaussian Naive Bayes algorithm
    gnb = GaussianNB()
    gnb.fit(X, y)

    if request.method == 'POST':
        # Get user input from the HTML form
        GWA = float(request.form['GWA'])
        preBoardLETscore = float(request.form['preBoardLETscore'])
        input_data = [[GWA, preBoardLETscore]]

        # Make a prediction using user input
        prediction = gnb.predict(input_data)[0]

        # Calculate the probability of the predicted outcome
        outcome_prob = gnb.predict_proba(input_data)[0]
        outcome_percentage = "{:.2%}".format(outcome_prob.max())

        # Group data by outcome and count occurrences
        data = df.groupby('outcome')['outcome'].count()

        # Plot a scatter plot of the data using Plotly
        fig = px.scatter(df, x='GWA', y='preBoardLETscore', color='outcome')

        # Generate bar chart of outcome occurrences
        chart = generate_bar_chart(data)

        # Render the HTML template with the prediction and chart
        return render_template('analyze.html', outcome=prediction, prediction=prediction, outcome_percentage=outcome_percentage, graphJSON=fig.to_json(), chart=chart, GWA=GWA, preBoardLETscore=preBoardLETscore)

    # Render the HTML template with empty form
    return render_template('analyze.html')


# Route for uploading a CSV file and analyzing student data in batches
@app.route('/upload', methods=['POST'])
def upload():

    # check if the post request has the file part
    if 'file' not in request.files:
        return render_template('home.html', message='No file selected')

    # get the file from the post request
    file = request.files['file']

    # if the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return render_template('home.html', message='No file selected')

    # read the CSV file
    data = pd.read_csv(file)

    # Load data from SQLite database
    df = pd.read_sql_query("SELECT * FROM datasets", conn)

    # Split the data into features and target variable
    X = df[['GWA', 'preBoardLETscore']]
    y = df['outcome']

    # Train the model using Gaussian Naive Bayes algorithm
    gnb = GaussianNB()
    gnb.fit(X, y)

    # extract the necessary features
    features = data[['GWA', 'preBoardLETscore']]

    # predict the classes and probabilities using the Gaussian Naive Bayes model
    predictions = gnb.predict(features)
    probabilities = gnb.predict_proba(features)

    # combine the features, predictions, and probabilities into a list of tuples
    features = features.values.tolist()
    predictions = predictions.tolist()
    probabilities = [max(p) for p in probabilities.tolist()]

    predictions_with_probabilities = list(zip(features, predictions, probabilities))

    # count the number of passed and failed students
    num_passed = sum(1 for p in predictions if p.lower() == 'passed')
    num_failed = sum(1 for p in predictions if p.lower() == 'failed')


    return render_template('batchresultanalysis.html',
                           predictions_with_probabilities=predictions_with_probabilities,
                           enumerate=enumerate,
                           num_passed=num_passed,
                           num_failed=num_failed)


# Define the route for displaying the evaluation metrics
@app.route('/metrics')
def metrics():
    # Check if the user is logged in
    if 'username' not in session:
        # If the user is not logged in, redirect them to the login page
        return redirect(url_for('login'))

    # Load data from SQLite database
    df = pd.read_sql_query("SELECT * FROM datasets", conn)

    # Split the data into features and target variable
    X = df[['GWA', 'preBoardLETscore']]
    y = df['outcome']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model using Gaussian Naive Bayes algorithm
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = gnb.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='PASSED')
    recall = recall_score(y_test, y_pred, pos_label='PASSED')
    f1 = f1_score(y_test, y_pred, pos_label='PASSED')

    # Calculate evaluation metrics as percentages instead of decimal numbers
    percentagesaccuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    percentagesprecision = round(precision_score(y_test, y_pred, pos_label='PASSED') * 100, 2)
    percentagesrecall = round(recall_score(y_test, y_pred, pos_label='PASSED') * 100, 2)
    percentagesf1 = round(f1_score(y_test, y_pred, pos_label='PASSED') * 100, 2)

    # Render the evaluation metrics template with the metric scores as variables
    return render_template('metrics.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1, percentagesaccuracy=percentagesaccuracy, percentagesprecision=percentagesprecision, percentagesrecall=percentagesrecall, percentagesf1=percentagesf1)


# Route for home page
@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html')
    else:
        return redirect(url_for('login'))

# Route for get started page
@app.route('/getstarted')
def getstarted():
    if 'username' in session:
        return render_template('getstarted.html')
    else:
        return redirect(url_for('login'))

# Route for about page
@app.route('/about')
def about():
    if 'username' in session:
        return render_template('about.html')
    else:
        return redirect(url_for('login'))

# Route for contact page
@app.route('/contact')
def contact():
    if 'username' in session:
        return render_template('contact.html')
    else:
        return redirect(url_for('login'))

# Route for documentation page
@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

# Route for uploading datasets
@app.route('/uploaddatasets')
def uploaddatasets():
    return render_template('uploaddatasets.html')

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        if row is not None and bcrypt.checkpw(password.encode('utf-8'), row[0]):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            error = 'Invalid Credentials. Please try again.'
        cursor.close()
        conn.close()
    return render_template('login.html', error=error)

# Route for sign up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        # Validate the reCAPTCHA response
        if not recaptcha.verify():
            error = 'reCAPTCHA verification failed.'
            return render_template('signup.html', error=error)

        # Process the form data
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            error = 'Username and password are required.'
        else:
            conn = sqlite3.connect('data.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            if user is not None:
                error = 'Username already taken.'
            else:
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
                conn.commit()
                cursor.close()
                conn.close()
                error = 'Credentials saved!'
                # return redirect(url_for('sucessreg'))

    # Render the signup page
    return render_template('signup.html', error=error)

# Route for logout page
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Route for sucess registration
@app.route('/sucessreg')
def sucessreg():
        return render_template('sucessreg.html')


# Running the app
if __name__ == "__main__":
     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

