from flask import Flask, render_template, request, jsonify
import pickle
import sqlite3
import pandas as pd
import numpy as np
import os
import re
import sys
import subprocess
from datetime import datetime

app = Flask(__name__)

# Security: Secret key and max request payload limit (1 MB) to prevent memory attacks
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

# 1. Auto-create & train models if missing on a brand-new server
if not os.path.exists('models/multi_health_models.pkl'):
    subprocess.run([sys.executable, 'train_model.py'], check=True)

with open('models/multi_health_models.pkl', 'rb') as f:
    models = pickle.load(f)

# 2. HTTP Security Headers (Protects against Clickjacking, XSS & MIME sniffing)
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# 3. Server-side Input Clamping & Sanitization (Never trust frontend inputs alone)
def clean_num(val, min_v, max_v, default):
    try:
        return max(min_v, min(max_v, float(val)))
    except (ValueError, TypeError):
        return default

def clean_name(name):
    if not name:
        return 'Suman Sarkar'
    # Strip any HTML/script characters; allow only letters, numbers, spaces, dots, and hyphens
    cleaned = re.sub(r'[^a-zA-Z0-9 .\-]', '', str(name)).strip()
    return cleaned[:40] if cleaned else 'Suman Sarkar'

# 4. Auto-create database AND seed starter timeline data
def init_db():
    conn = sqlite3.connect('medisense.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS assessments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_name TEXT,
                  timestamp TEXT,
                  health_score INTEGER,
                  diabetes_risk REAL,
                  heart_risk REAL,
                  kidney_risk REAL)''')
    try:
        c.execute("ALTER TABLE assessments ADD COLUMN patient_name TEXT DEFAULT 'Suman Sarkar'")
    except sqlite3.OperationalError:
        pass

    c.execute("SELECT COUNT(*) FROM assessments")
    if c.fetchone()[0] == 0:
        demo_rows = [
            ('Suman Sarkar', '01 Sep', 64, 48.0, 42.0, 35.0),
            ('Suman Sarkar', '01 Oct', 72, 36.0, 31.0, 28.0),
            ('Suman Sarkar', '01 Nov', 81, 22.0, 19.0, 15.0)
        ]
        c.executemany("""INSERT INTO assessments 
                         (patient_name, timestamp, health_score, diabetes_risk, heart_risk, kidney_risk) 
                         VALUES (?, ?, ?, ?, ?, ?)""", demo_rows)

    conn.commit()
    conn.close()

init_db()

# 5. Core ML evaluation & Explainable AI helper function
def evaluate_health(age, bmi, bp, glucose, sleep, activity):
    features = ['Age', 'BMI', 'BloodPressure', 'Glucose', 'SleepHours', 'PhysicalActivity']
    input_df = pd.DataFrame([[age, bmi, bp, glucose, sleep, activity]], columns=features)

    d_prob = round(models['diabetes'].predict_proba(input_df)[0][1] * 100, 1)
    h_prob = round(models['heart'].predict_proba(input_df)[0][1] * 100, 1)
    k_prob = round(models['kidney'].predict_proba(input_df)[0][1] * 100, 1)

    avg_risk = (d_prob + h_prob + k_prob) / 3
    health_score = max(10, round(100 - (avg_risk * 0.75)))

    importances = np.zeros(len(features))
    for key in ['diabetes', 'heart', 'kidney']:
        m = models[key]
        if hasattr(m, 'feature_importances_'):
            importances += m.feature_importances_ / 3.0
        else:
            importances += np.abs(m.coef_[0]) / np.sum(np.abs(m.coef_[0])) / 3.0

    deviations = {
        'Glucose': max(0.1, (glucose - 90) / 70.0) * importances[3],
        'BMI': max(0.1, (bmi - 21.5) / 12.0) * importances[1],
        'Blood Pressure': max(0.1, (bp - 110) / 45.0) * importances[2],
        'Sleep Deficit': max(0.1, (7.5 - sleep) / 3.0) * importances[4],
        'Inactivity': (1.0 if activity == 0 else 0.15) * max(0.08, importances[5])
    }
    max_dev = max(deviations.values()) if max(deviations.values()) > 0 else 1.0
    impacts = {k: min(100, max(12, round((v / max_dev) * 95))) for k, v in deviations.items()}

    return health_score, d_prob, h_prob, k_prob, impacts

# 6. Main Dashboard Route
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    patient_name = clean_name(request.form.get('patient_name', 'Suman Sarkar'))

    if request.method == 'POST':
        age = clean_num(request.form.get('age'), 1, 120, 45)
        bmi = clean_num(request.form.get('bmi'), 10.0, 65.0, 25.0)
        bp = clean_num(request.form.get('bp'), 60, 250, 120)
        glucose = clean_num(request.form.get('glucose'), 40, 400, 100)
        sleep = clean_num(request.form.get('sleep'), 1.0, 16.0, 7.0)
        activity = 1 if str(request.form.get('activity')) == '1' else 0

        health_score, d_prob, h_prob, k_prob, impacts = evaluate_health(age, bmi, bp, glucose, sleep, activity)

        conn = sqlite3.connect('medisense.db')
        c = conn.cursor()
        # Parameterized SQL (?) prevents SQL Injection attacks
        c.execute("""INSERT INTO assessments 
                     (patient_name, timestamp, health_score, diabetes_risk, heart_risk, kidney_risk) 
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (patient_name, datetime.now().strftime("%d %b %H:%M"), health_score, d_prob, h_prob, k_prob))
        conn.commit()
        conn.close()

        result = {
            'patient_name': patient_name,
            'age': age, 'bmi': bmi, 'bp': bp, 'glucose': glucose, 'sleep': sleep, 'activity': activity,
            'health_score': health_score,
            'diabetes_risk': d_prob,
            'heart_risk': h_prob,
            'kidney_risk': k_prob,
            'impacts': impacts
        }

    conn = sqlite3.connect('medisense.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, health_score, diabetes_risk, heart_risk FROM assessments WHERE patient_name = ? ORDER BY id DESC LIMIT 8", (patient_name,))
    history = c.fetchall()[::-1]
    conn.close()

    return render_template('index.html', result=result, history=history, patient_name=patient_name)

# 7. Live API endpoint for the interactive What-If Sliders
@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json(silent=True) or {}
    score, d_prob, h_prob, k_prob, _ = evaluate_health(
        clean_num(data.get('age'), 1, 120, 45),
        clean_num(data.get('bmi'), 10.0, 65.0, 25.0),
        clean_num(data.get('bp'), 60, 250, 120),
        clean_num(data.get('glucose'), 40, 400, 100),
        clean_num(data.get('sleep'), 1.0, 16.0, 7.0),
        1 if str(data.get('activity')) == '1' else 0
    )
    return jsonify({
        'health_score': score,
        'diabetes_risk': d_prob,
        'heart_risk': h_prob,
        'kidney_risk': k_prob
    })

if __name__ == '__main__':
    # Automatically turns OFF debug mode on cloud servers while allowing local testing
    is_local = os.environ.get('RENDER') is None
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=is_local)