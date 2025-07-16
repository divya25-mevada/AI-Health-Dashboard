from flask import Flask, render_template, request, redirect, session, flash, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
from chatbot import ai_chatbot
import plotly.graph_objs as go
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'your_password'
app.config['MYSQL_DB'] = 'healthapp'

mysql = MySQL(app)

# Load ML Model
MODEL_PATH = 'model/trained_model.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None

# Home
@app.route('/')
def home():
    return redirect('/login')

# Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = request.form['username']
        pw = request.form['password']
        
        try:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute("SELECT * FROM users WHERE username = %s", (user,))
            account = cursor.fetchone()
            
            if account:
                flash('Account already exists!')
                return render_template('register.html')
            else:
                hashed_pw = generate_password_hash(pw)
                cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (user, hashed_pw))
                mysql.connection.commit()
                cursor.close()
                flash('Registered successfully. You can now log in.')
                return redirect('/login')
        except Exception as e:
            flash(f'Database error: {str(e)}')
            return render_template('register.html')
    
    return render_template('register.html')

# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pw = request.form['password']
        
        try:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute("SELECT * FROM users WHERE username = %s", (user,))
            account = cursor.fetchone()
            cursor.close()
            
            if account and check_password_hash(account['password'], pw):
                session['loggedin'] = True
                session['id'] = account['id']
                session['username'] = account['username']
                session['chat_history'] = []
                return redirect('/dashboard')
            else:
                flash('Incorrect username or password')
        except Exception as e:
            flash(f'Database error: {str(e)}')
    
    return render_template('login.html')

# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.')
    return redirect('/login')

# Train Model (Admin/Doctor only)
@app.route('/train', methods=['POST'])
def train():
    if 'loggedin' not in session:
        return redirect('/login')
    try:
        os.system("python train_model.py")
        global model
        model = joblib.load(MODEL_PATH)
        flash("✅ Model retrained successfully!")
    except Exception as e:
        flash(f"⚠️ Error training model: {str(e)}")
    return redirect('/dashboard')

# Dashboard
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'loggedin' not in session:
        return redirect('/login')

    if request.method == 'POST':
        if model is None:
            flash('Model not loaded. Please contact administrator.')
            return render_template("dashboard.html", chat=session.get('chat_history', []))
        
        try:
            # Get form data
            age = int(request.form['age'])
            bmi = float(request.form['bmi'])
            bp = int(request.form['bp'])
            glucose = int(request.form['glucose'])
            insulin = int(request.form['insulin'])
            preg = int(request.form['preg'])

            # Create input data for prediction (must match training order)
            input_data = pd.DataFrame([[preg, glucose, bp, insulin, bmi, age]],
                                      columns=["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "Age"])
            
            print(f"🔍 Input data for prediction: {input_data.iloc[0].to_dict()}")  # Debug
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            print(f"🎯 Prediction: {prediction}, Probability: {prediction_proba}")  # Debug
            
            # Save to database
            cursor = mysql.connection.cursor()
            cursor.execute("""
                INSERT INTO predictions (user_id, age, bmi, blood_pressure, glucose, insulin, pregnancies, result)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (session['id'], age, bmi, bp, glucose, insulin, preg, int(prediction)))
            mysql.connection.commit()
            cursor.close()

            # Prepare result data
            result_data = {
                'prediction': int(prediction),
                'risk_score': calculate_risk_score(age, bmi, bp, glucose, insulin, preg),
                'bmi_category': get_bmi_category(bmi),
                'glucose_status': get_glucose_status(glucose),
                'bp_status': get_bp_status(bp),
                'health_tips': generate_health_tips(age, bmi, bp, glucose, insulin, preg, prediction)
            }

            return render_template("dashboard.html", 
                                   result=result_data,
                                   chat=session.get('chat_history', []),
                                   username=session['username'])

        except ValueError as e:
            flash(f'Invalid input: Please enter valid numbers')
            return render_template("dashboard.html", chat=session.get('chat_history', []))
        except Exception as e:
            flash(f'Error: {str(e)}')
            return render_template("dashboard.html", chat=session.get('chat_history', []))

    # GET: Show recent predictions
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM predictions WHERE user_id = %s ORDER BY predicted_at DESC LIMIT 10", (session['id'],))
        recent_predictions = cursor.fetchall()
        cursor.close()
    except Exception as e:
        flash(f'Error loading predictions: {str(e)}')
        recent_predictions = []

    return render_template("dashboard.html",
                           chat=session.get('chat_history', []),
                           recent_predictions=recent_predictions,
                           username=session['username'])

# Chatbot (Gemini) - FIXED
@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'loggedin' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
            
        user_input = data['message']
        user_name = session.get('username', 'User')

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        # Fetch last 5 predictions for context
        cursor.execute("""
            SELECT age, bmi, blood_pressure, glucose, insulin, result, predicted_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY predicted_at DESC
            LIMIT 5
        """, (session['id'],))
        health_history = cursor.fetchall()

        # Fetch only latest prediction for accuracy
        cursor.execute("""
            SELECT age, bmi, blood_pressure, glucose, insulin, result, predicted_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY predicted_at DESC
            LIMIT 1
        """, (session['id'],))
        latest_data = cursor.fetchone()
        cursor.close()

        # Get AI reply using history
        bot_reply = ai_chatbot(user_input, user_name, health_history, latest_data)

        # Store in session
        if 'chat_history' not in session:
            session['chat_history'] = []
        session['chat_history'].append((user_input, bot_reply))

        return jsonify({'response': bot_reply})
        
    except Exception as e:
        return jsonify({'error': f'Chat error: {str(e)}'}), 500

# Prediction History
@app.route('/history')
def history():
    if 'loggedin' not in session:
        return redirect('/login')

    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM predictions WHERE user_id = %s ORDER BY predicted_at DESC", (session['id'],))
        data = cursor.fetchall()
        cursor.close()

        charts = {}
        if data:
            dates = [row['predicted_at'].strftime('%Y-%m-%d') for row in data]
            glucose = [row['glucose'] for row in data]
            bmi = [row['bmi'] for row in data]
            bp = [row['blood_pressure'] for row in data]
            risk = [row['result'] for row in data]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=glucose, name='Glucose', mode='lines+markers', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dates, y=bmi, name='BMI', mode='lines+markers', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=dates, y=bp, name='Blood Pressure', mode='lines+markers', line=dict(color='blue')))
            fig.update_layout(title='Health Trends', xaxis_title='Date', hovermode='x unified')

            charts['trend'] = fig.to_html(full_html=False)

        return render_template("history.html", data=data, charts=charts)
    
    except Exception as e:
        flash(f'Error loading history: {str(e)}')
        return render_template("history.html", data=[], charts={})

# Helper functions
def calculate_risk_score(age, bmi, bp, glucose, insulin, pregnancies):
    score = 0
    if age > 45: score += 20
    elif age > 35: score += 10
    if bmi > 30: score += 25
    elif bmi > 25: score += 15
    if bp > 140: score += 20
    elif bp > 120: score += 10
    if glucose > 126: score += 30
    elif glucose > 100: score += 15
    if insulin > 166: score += 15
    if pregnancies > 3: score += 10
    return min(score, 100)

def get_bmi_category(bmi):
    if bmi < 18.5: return 'Underweight'
    elif bmi < 25: return 'Normal'
    elif bmi < 30: return 'Overweight'
    return 'Obese'

def get_glucose_status(glucose):
    if glucose < 70: return 'Low'
    elif glucose < 100: return 'Normal'
    elif glucose < 126: return 'Pre-diabetic'
    return 'Diabetic'

def get_bp_status(bp):
    if bp < 90: return 'Low'
    elif bp < 120: return 'Normal'
    elif bp < 140: return 'High Normal'
    return 'High'

def generate_health_tips(age, bmi, bp, glucose, insulin, preg, prediction):
    tips = []
    if prediction == 1:
        tips.append("⚠️ High risk detected. Please consult your doctor.")
    if bmi > 30:
        tips.append("🏃 Try regular exercise and a healthy diet.")
    if glucose > 126:
        tips.append("🩺 Monitor your glucose levels.")
    if bp > 140:
        tips.append("💊 Keep your blood pressure under control.")
    if insulin > 166:
        tips.append("🍽 Consider managing insulin with diet and guidance.")
    if age > 45:
        tips.append("🧬 Regular health checkups are important at your age.")
    if not tips:
        tips.append("✅ Keep up the good health!")
    return " ".join(tips)

# Database connection test
@app.route('/test_db')
def test_db():
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        return jsonify({'status': 'success', 'message': 'Database connection working'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# Run App
if __name__ == "__main__":
    app.run(debug=True)
