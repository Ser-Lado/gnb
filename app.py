from flask import Flask, request, render_template, redirect, session, url_for

from sklearn.naive_bayes import GaussianNB
import sqlite3
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import bcrypt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split




# This code is to generate dummy data sets. ###################################################################
#  import random
#     # Create a connection and cursor object
#     conn = sqlite3.connect('data.db')
#     cursor = conn.cursor()

#     # Create the datasets table if it doesn't exist
#     cursor.execute("CREATE TABLE IF NOT EXISTS datasets (GWA REAL, preBoardLETscore REAL, outcome TEXT)")

#     # Generate 1000 random values and insert them into the datasets table
#     for i in range(3000):
#         gwa = round(random.uniform(1.0, 3.0), 1)
#         preboard_let_score = round(random.uniform(70.0, 100.0), 1)
#         outcome = random.choice(['Passed', 'Failed'])
#         cursor.execute("INSERT INTO datasets VALUES (?, ?, ?)", (gwa, preboard_let_score, outcome))

#     # Commit the changes and close the connection
#     conn.commit()
#     conn.close()

#     print("Successfully inserted 1000 rows into the datasets table!")


# Declare a Flask app
app = Flask(__name__) #Initialize the flask App
app.secret_key = 'secret_key'

# SQLite database connection for GRAPH
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()


# Create the users table if it does not exist
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()


# This code is for BAR Graph - STRART ************
# function to get data from SQLite database
def get_data_from_db():
    conn = sqlite3.connect('data.db', check_same_thread=False)
    df = pd.read_sql_query("SELECT * from datasets", conn)
    conn.close()
    return df

# function to preprocess data and train model
def preprocess_and_train_model(df):
    # add preprocessing code here
    # ...
    # train model using Gaussian Naive Bayes
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
    

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    gnb = train_model()
    GWA = request.form['GWA']
    preBoardLETscore = request.form['preBoardLETscore']
    input_data = [[float(GWA), float(preBoardLETscore)]]
    outcome = gnb.predict(input_data)[0]

    # code that retrieves the values of "GWA" and "preBoardLETscore"
    gwa = request.args.get('GWA')
    preBoardLETscore = request.args.get('preBoardLETscore')
   


    #for graph ############################################
    # Get user input from the HTML form
    GWA = float(request.form['GWA'])
    preBoardLETscore = float(request.form['preBoardLETscore'])
    
    # Load data from SQLite database
    df = pd.read_sql_query("SELECT * FROM datasets", conn)

    # Split the data into features and target variable
    X = df[['GWA', 'preBoardLETscore']]
    y = df['outcome']

    # Train the model using Gaussian Naive Bayes algorithm
    gnb = GaussianNB()
    gnb.fit(X, y)

    # Make a prediction using user input
    prediction_input = pd.DataFrame([[GWA, preBoardLETscore]], columns=X.columns)
    prediction = gnb.predict(prediction_input)

    # Load data from SQLite database
    df = pd.read_sql_query("SELECT * FROM datasets", conn)

    # Plot a scatter plot of the data using Plotly
    fig = px.scatter(df, x='GWA', y='preBoardLETscore', color='outcome')

    # get data from database
    df = get_data_from_db()
    # preprocess data and train model
    model = preprocess_and_train_model(df)
    # handle form submission   
    if request.method == 'POST':
        # get user input
        gwa = float(request.form.get('GWA'))
        preboard = float(request.form.get('preBoardLETscore'))
        # make prediction using trained model
        prediction = model.predict([[gwa, preboard]])
        # group data by prediction and count occurrences
        data = df.groupby('outcome')['outcome'].count()
        # generate bar chart
        chart = generate_bar_chart(data)
      
    

    # Display the prediction in the HTML page
    return render_template('analyze.html', outcome=outcome, prediction=prediction[0], graphJSON=fig.to_json(), chart=chart,  GWA=GWA, preBoardLETscore=preBoardLETscore)


    # return render_template('predict.html', outcome=outcome)



# Define the route for displaying the evaluation metrics
@app.route('/metrics')
def metrics():
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



# Navigations ################################333333
@app.route('/getstarted')
def getstarted():
    if 'username' in session:
        return render_template('getstarted.html')
    else:
        return redirect(url_for('login'))
    
@app.route('/about')
def about():
    if 'username' in session:
        return render_template('about.html')
    else:
        return redirect(url_for('login'))

@app.route('/contact')
def contact():
    if 'username' in session:
        return render_template('contact.html')
    else:
        return redirect(url_for('login'))

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/uploaddatasets')
def uploaddatasets():
    return render_template('uploaddatasets.html')

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

@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html')
    else:
        return redirect(url_for('login'))


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


# Route for sign up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
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
            # return redirect(url_for('login'))
    return render_template('signup.html', error=error)



# Running the app
if __name__ == "__main__":
     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

