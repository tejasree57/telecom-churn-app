"""
=============================================================================
TELECOM CHURN PREDICTION — WEB APPLICATION
Flask app with:
  - Login / logout (username + password)
  - Single customer prediction form
  - Batch CSV upload prediction
  - Model performance dashboard
Run locally  : python app.py
Deploy       : See deployment guide (Heroku / Render / Databricks Serving)
=============================================================================
"""

from flask import (Flask, render_template_string, request,
                   redirect, url_for, session, flash, jsonify, send_file)
from functools import wraps
import pandas as pd
import numpy as np
import pickle
import json
import io
import os

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "TelecomChurn@SecretKey#2024"   # Change in production!

# ─────────────────────────────────────────────
# User Credentials (hardcoded for demo)
# In production: use a database + hashed passwords
# ─────────────────────────────────────────────
USERS = {
    "admin":    "Admin@123",
    "analyst":  "Analyst@456",
    "manager":  "Manager@789",
}

# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
MODEL_PATH        = "churn_model.pkl"
FEATURE_PATH      = "feature_names.json"

def load_model():
    """Load pickled sklearn pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURE_PATH, "r") as f:
        features = json.load(f)
    return model, features

MODEL, FEATURES = load_model()

# ─────────────────────────────────────────────
# Login Required Decorator
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# HTML TEMPLATES (inline for single-file deploy)
# ─────────────────────────────────────────────

BASE_CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; }
  body { background: #f0f4f8; color: #333; }
  .navbar {
    background: linear-gradient(135deg, #1565C0, #0D47A1);
    padding: 14px 30px; color: white;
    display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }
  .navbar a { color: white; text-decoration: none; margin-left: 20px; font-size: 14px; }
  .navbar a:hover { text-decoration: underline; }
  .container { max-width: 960px; margin: 40px auto; padding: 0 20px; }
  .card {
    background: white; border-radius: 12px; padding: 32px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1); margin-bottom: 28px;
  }
  h1 { font-size: 26px; color: #1565C0; margin-bottom: 8px; }
  h2 { font-size: 20px; color: #1976D2; margin-bottom: 16px; }
  .subtitle { color: #666; margin-bottom: 24px; font-size: 14px; }
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .form-group { display: flex; flex-direction: column; gap: 6px; }
  .form-group label { font-size: 13px; font-weight: 600; color: #555; }
  .form-group input, .form-group select {
    padding: 10px 12px; border: 1px solid #ddd; border-radius: 8px;
    font-size: 14px; transition: border 0.2s;
  }
  .form-group input:focus, .form-group select:focus {
    outline: none; border-color: #1976D2;
  }
  .btn {
    padding: 12px 28px; border: none; border-radius: 8px; cursor: pointer;
    font-size: 15px; font-weight: 600; transition: all 0.2s;
  }
  .btn-primary { background: #1976D2; color: white; }
  .btn-primary:hover { background: #1565C0; transform: translateY(-1px); }
  .btn-danger  { background: #E53935; color: white; }
  .btn-success { background: #43A047; color: white; }
  .result-box {
    padding: 20px; border-radius: 10px; text-align: center;
    font-size: 20px; font-weight: bold; margin-top: 20px;
  }
  .churn-yes { background: #FFEBEE; color: #C62828; border: 2px solid #EF9A9A; }
  .churn-no  { background: #E8F5E9; color: #1B5E20; border: 2px solid #A5D6A7; }
  .alert { padding: 12px 18px; border-radius: 8px; margin-bottom: 16px; font-size: 14px; }
  .alert-error   { background: #FFEBEE; color: #C62828; }
  .alert-success { background: #E8F5E9; color: #2E7D32; }
  .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .stat-card {
    text-align: center; padding: 20px; border-radius: 10px;
    background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
  }
  .stat-value { font-size: 28px; font-weight: bold; color: #1565C0; }
  .stat-label { font-size: 13px; color: #555; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }
  th { background: #1565C0; color: white; padding: 10px 14px; text-align: left; }
  td { padding: 10px 14px; border-bottom: 1px solid #eee; }
  tr:hover td { background: #f5f5f5; }
  .badge {
    padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;
  }
  .badge-yes { background: #FFCDD2; color: #B71C1C; }
  .badge-no  { background: #C8E6C9; color: #1B5E20; }
  .login-wrap {
    min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
  }
  .login-card {
    background: white; border-radius: 16px; padding: 44px 40px;
    width: 380px; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
  }
  .login-logo { text-align: center; margin-bottom: 24px; }
  .login-logo h1 { font-size: 22px; color: #1565C0; margin-top: 8px; }
  .login-logo p  { font-size: 13px; color: #888; }
</style>
"""

# ─────────────────────────────────────────────
# ROUTE: Login
# ─────────────────────────────────────────────
LOGIN_HTML = BASE_CSS + """
<div class="login-wrap">
  <div class="login-card">
    <div class="login-logo">
      <div style="font-size:48px">📡</div>
      <h1>Telecom Churn AI</h1>
      <p>Sign in to access the dashboard</p>
    </div>
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% for category, message in messages %}
        <div class="alert alert-{{ category }}">{{ message }}</div>
      {% endfor %}
    {% endwith %}
    <form method="POST">
      <div class="form-group" style="margin-bottom:16px">
        <label>Username</label>
        <input type="text" name="username" placeholder="Enter username" required autofocus>
      </div>
      <div class="form-group" style="margin-bottom:24px">
        <label>Password</label>
        <input type="password" name="password" placeholder="Enter password" required>
      </div>
      <button class="btn btn-primary" style="width:100%" type="submit">Sign In →</button>
    </form>
    <p style="margin-top:20px;font-size:12px;color:#999;text-align:center">
      Demo accounts: admin / analyst / manager
    </p>
  </div>
</div>
"""

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        if u in USERS and USERS[u] == p:
            session["username"] = u
            return redirect(url_for("dashboard"))
        flash("Invalid username or password", "error")
    return render_template_string(LOGIN_HTML)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─────────────────────────────────────────────
# ROUTE: Dashboard (Home)
# ─────────────────────────────────────────────
DASH_HTML = BASE_CSS + """
<div class="navbar">
  <span style="font-size:18px;font-weight:bold">📡 Telecom Churn AI</span>
  <div>
    <a href="{{ url_for('predict_single') }}">🔍 Predict</a>
    <a href="{{ url_for('predict_batch') }}">📂 Batch Upload</a>
    <a href="{{ url_for('dashboard') }}">🏠 Dashboard</a>
    <a href="{{ url_for('logout') }}">🚪 Logout ({{ user }})</a>
  </div>
</div>
<div class="container">
  <div class="card">
    <h1>Welcome back, {{ user }}! 👋</h1>
    <p class="subtitle">Telecom Customer Churn Prediction System — Powered by Logistic Regression</p>
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-value">7,043</div>
        <div class="stat-label">Training Customers</div>
      </div>
      <div class="stat-card" style="background:linear-gradient(135deg,#E8F5E9,#C8E6C9)">
        <div class="stat-value" style="color:#2E7D32">~82%</div>
        <div class="stat-label">Model Accuracy</div>
      </div>
      <div class="stat-card" style="background:linear-gradient(135deg,#FFF3E0,#FFE0B2)">
        <div class="stat-value" style="color:#E65100">~84%</div>
        <div class="stat-label">ROC-AUC Score</div>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Quick Actions</h2>
    <div style="display:flex;gap:16px;flex-wrap:wrap">
      <a href="{{ url_for('predict_single') }}">
        <button class="btn btn-primary">🔍 Predict Single Customer</button>
      </a>
      <a href="{{ url_for('predict_batch') }}">
        <button class="btn btn-success">📂 Upload CSV for Batch Prediction</button>
      </a>
    </div>
  </div>
  <div class="card">
    <h2>About This Model</h2>
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Algorithm</td><td>Logistic Regression (sklearn)</td></tr>
      <tr><td>Features Used</td><td>21 customer attributes</td></tr>
      <tr><td>Train/Test Split</td><td>80% / 20% (stratified)</td></tr>
      <tr><td>Cross-Validation</td><td>5-fold Stratified KFold</td></tr>
      <tr><td>Class Handling</td><td>balanced class weights</td></tr>
      <tr><td>Regularization</td><td>L2 (C=1.0)</td></tr>
    </table>
  </div>
</div>
"""

@app.route("/")
@login_required
def dashboard():
    return render_template_string(DASH_HTML, user=session["username"])

# ─────────────────────────────────────────────
# ROUTE: Single Customer Prediction
# ─────────────────────────────────────────────
PREDICT_HTML = BASE_CSS + """
<div class="navbar">
  <span style="font-size:18px;font-weight:bold">📡 Telecom Churn AI</span>
  <div>
    <a href="{{ url_for('dashboard') }}">🏠 Dashboard</a>
    <a href="{{ url_for('predict_batch') }}">📂 Batch Upload</a>
    <a href="{{ url_for('logout') }}">🚪 Logout</a>
  </div>
</div>
<div class="container">
  <div class="card">
    <h1>🔍 Predict Customer Churn</h1>
    <p class="subtitle">Fill in customer details to get a churn probability prediction</p>
    <form method="POST">
      <div class="form-grid">
        <div class="form-group">
          <label>Gender</label>
          <select name="gender">
            <option value="1">Male</option>
            <option value="0">Female</option>
          </select>
        </div>
        <div class="form-group">
          <label>Senior Citizen</label>
          <select name="SeniorCitizen">
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>
        <div class="form-group">
          <label>Has Partner</label>
          <select name="Partner">
            <option value="1">Yes</option><option value="0">No</option>
          </select>
        </div>
        <div class="form-group">
          <label>Has Dependents</label>
          <select name="Dependents">
            <option value="0">No</option><option value="1">Yes</option>
          </select>
        </div>
        <div class="form-group">
          <label>Tenure (months)</label>
          <input type="number" name="tenure" value="12" min="0" max="72">
        </div>
        <div class="form-group">
          <label>Phone Service</label>
          <select name="PhoneService">
            <option value="1">Yes</option><option value="0">No</option>
          </select>
        </div>
        <div class="form-group">
          <label>Multiple Lines</label>
          <select name="MultipleLines">
            <option value="0">No</option><option value="1">Yes</option>
          </select>
        </div>
        <div class="form-group">
          <label>Online Security</label>
          <select name="OnlineSecurity">
            <option value="0">No</option><option value="1">Yes</option>
          </select>
        </div>
        <div class="form-group">
          <label>Online Backup</label>
          <select name="OnlineBackup">
            <option value="0">No</option><option value="1">Yes</option>
          </select>
        </div>
        <div class="form-group">
          <label>Device Protection</label>
          <select name="DeviceProtection">
            <option value="0">No</option><option value="1">Yes</option>
          </select>
        </div>
        <div class="form-group">
          <label>Tech Support</label>
          <select name="TechSupport">
            <option value="0">No</option><option value="1">Yes</option>
          </select>
        </div>
        <div class="form-group">
          <label>Streaming TV</label>
          <select name="StreamingTV">
            <option value="0">No</option><option value="1">Yes</option>
          </select>
        </div>
        <div class="form-group">
          <label>Streaming Movies</label>
          <select name="StreamingMovies">
            <option value="0">No</option><option value="1">Yes</option>
          </select>
        </div>
        <div class="form-group">
          <label>Paperless Billing</label>
          <select name="PaperlessBilling">
            <option value="1">Yes</option><option value="0">No</option>
          </select>
        </div>
        <div class="form-group">
          <label>Monthly Charges ($)</label>
          <input type="number" name="MonthlyCharges" value="65.00" step="0.01">
        </div>
        <div class="form-group">
          <label>Total Charges ($)</label>
          <input type="number" name="TotalCharges" value="780.00" step="0.01">
        </div>
        <div class="form-group">
          <label>Admin Tickets Raised</label>
          <input type="number" name="numAdminTickets" value="0" min="0">
        </div>
        <div class="form-group">
          <label>Tech Tickets Raised</label>
          <input type="number" name="numTechTickets" value="0" min="0">
        </div>
        <div class="form-group">
          <label>Internet Service</label>
          <select name="InternetService">
            <option value="Fiber optic">Fiber optic</option>
            <option value="DSL">DSL</option>
            <option value="No">No Internet</option>
          </select>
        </div>
        <div class="form-group">
          <label>Contract Type</label>
          <select name="Contract">
            <option value="Month-to-month">Month-to-month</option>
            <option value="One year">One year</option>
            <option value="Two year">Two year</option>
          </select>
        </div>
        <div class="form-group">
          <label>Payment Method</label>
          <select name="PaymentMethod">
            <option value="Electronic check">Electronic check</option>
            <option value="Mailed check">Mailed check</option>
            <option value="Bank transfer (automatic)">Bank transfer</option>
            <option value="Credit card (automatic)">Credit card</option>
          </select>
        </div>
      </div>
      <br>
      <button class="btn btn-primary" type="submit">⚡ Predict Churn Risk</button>
    </form>

    {% if result is not none %}
      <div class="result-box {{ 'churn-yes' if result.churn else 'churn-no' }}">
        {% if result.churn %}
          ⚠️ HIGH CHURN RISK — Probability: {{ result.prob }}%
        {% else %}
          ✅ LOW CHURN RISK — Probability: {{ result.prob }}%
        {% endif %}
      </div>
      <p style="text-align:center;margin-top:10px;color:#666;font-size:13px">
        Prediction threshold: 50% | Model confidence: {{ result.prob }}%
      </p>
    {% endif %}
  </div>
</div>
"""

def build_input_vector(form, features):
    """Convert form data → numpy array aligned to model features."""
    # One-hot encode categoricals to match training
    internet = form.get("InternetService", "Fiber optic")
    contract  = form.get("Contract", "Month-to-month")
    payment   = form.get("PaymentMethod", "Electronic check")

    raw = {
        "gender":           float(form.get("gender", 1)),
        "SeniorCitizen":    float(form.get("SeniorCitizen", 0)),
        "Partner":          float(form.get("Partner", 0)),
        "Dependents":       float(form.get("Dependents", 0)),
        "tenure":           float(form.get("tenure", 12)),
        "PhoneService":     float(form.get("PhoneService", 1)),
        "MultipleLines":    float(form.get("MultipleLines", 0)),
        "OnlineSecurity":   float(form.get("OnlineSecurity", 0)),
        "OnlineBackup":     float(form.get("OnlineBackup", 0)),
        "DeviceProtection": float(form.get("DeviceProtection", 0)),
        "TechSupport":      float(form.get("TechSupport", 0)),
        "StreamingTV":      float(form.get("StreamingTV", 0)),
        "StreamingMovies":  float(form.get("StreamingMovies", 0)),
        "PaperlessBilling": float(form.get("PaperlessBilling", 1)),
        "MonthlyCharges":   float(form.get("MonthlyCharges", 65)),
        "TotalCharges":     float(form.get("TotalCharges", 780)),
        "numAdminTickets":  float(form.get("numAdminTickets", 0)),
        "numTechTickets":   float(form.get("numTechTickets", 0)),
        # One-hot encoded columns (drop_first=True mirrors training)
        "InternetService_Fiber optic": 1.0 if internet == "Fiber optic" else 0.0,
        "InternetService_No":          1.0 if internet == "No"           else 0.0,
        "Contract_One year":           1.0 if contract  == "One year"    else 0.0,
        "Contract_Two year":           1.0 if contract  == "Two year"    else 0.0,
        "PaymentMethod_Credit card (automatic)":  1.0 if payment == "Credit card (automatic)"  else 0.0,
        "PaymentMethod_Electronic check":         1.0 if payment == "Electronic check"         else 0.0,
        "PaymentMethod_Mailed check":             1.0 if payment == "Mailed check"             else 0.0,
    }
    row = [raw.get(f, 0.0) for f in features]
    return np.array(row).reshape(1, -1)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict_single():
    result = None
    if request.method == "POST" and MODEL is not None:
        X_input = build_input_vector(request.form, FEATURES)
        prob    = MODEL.predict_proba(X_input)[0][1] * 100
        churn   = prob >= 50.0
        result  = {"prob": round(prob, 1), "churn": churn}
    return render_template_string(PREDICT_HTML, result=result)

# ─────────────────────────────────────────────
# ROUTE: Batch CSV Prediction
# ─────────────────────────────────────────────
BATCH_HTML = BASE_CSS + """
<div class="navbar">
  <span style="font-size:18px;font-weight:bold">📡 Telecom Churn AI</span>
  <div>
    <a href="{{ url_for('dashboard') }}">🏠 Dashboard</a>
    <a href="{{ url_for('predict_single') }}">🔍 Predict</a>
    <a href="{{ url_for('logout') }}">🚪 Logout</a>
  </div>
</div>
<div class="container">
  <div class="card">
    <h1>📂 Batch Prediction — Upload CSV</h1>
    <p class="subtitle">Upload your customer CSV file. Must contain the same columns as the training data (excluding customerID and Churn).</p>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv" required style="margin-bottom:16px;display:block">
      <button class="btn btn-primary" type="submit">⚡ Run Predictions</button>
    </form>
    {% if table_html %}
      <br>
      <h2>Results (first 50 rows shown)</h2>
      {{ table_html | safe }}
      <br>
      <a href="{{ url_for('download_results') }}">
        <button class="btn btn-success">⬇️ Download Full Results CSV</button>
      </a>
    {% endif %}
  </div>
</div>
"""

batch_results_cache = {}   # simple in-memory cache per session

@app.route("/batch", methods=["GET", "POST"])
@login_required
def predict_batch():
    table_html = None
    if request.method == "POST":
        f = request.files.get("file")
        if f and MODEL is not None:
            df = pd.read_csv(f)
            df_clean = df.copy()
            # Apply same encoding as training
            binary_cols = ["Partner","Dependents","PhoneService","PaperlessBilling",
                           "MultipleLines","OnlineSecurity","OnlineBackup",
                           "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
            for col in binary_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].map(
                        {"Yes":1,"No":0,"No phone service":0,"No internet service":0}
                    ).fillna(0)
            if "gender" in df_clean.columns:
                df_clean["gender"] = df_clean["gender"].map({"Male":1,"Female":0}).fillna(0)
            ohe_cols = ["InternetService","Contract","PaymentMethod"]
            df_clean = pd.get_dummies(df_clean, columns=ohe_cols, drop_first=True)
            if "TotalCharges" in df_clean.columns:
                df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
                df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median(), inplace=True)
            for col in FEATURES:
                if col not in df_clean.columns:
                    df_clean[col] = 0
            X_batch   = df_clean[FEATURES].values
            probs     = MODEL.predict_proba(X_batch)[:, 1] * 100
            df["Churn_Probability_%"] = probs.round(1)
            df["Churn_Prediction"]    = ["Yes" if p >= 50 else "No" for p in probs]
            batch_results_cache[session["username"]] = df
            display_df = df.head(50).copy()
            display_df["Churn_Prediction"] = display_df["Churn_Prediction"].apply(
                lambda x: f'<span class="badge badge-{"yes" if x=="Yes" else "no"}">{x}</span>'
            )
            table_html = display_df.to_html(index=False, escape=False,
                                            classes="", border=0)
    return render_template_string(BATCH_HTML, table_html=table_html)

@app.route("/download_results")
@login_required
def download_results():
    df = batch_results_cache.get(session["username"])
    if df is None:
        return redirect(url_for("predict_batch"))
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.read().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="churn_predictions.csv"
    )

# ─────────────────────────────────────────────
# Run App
# ─────────────────────────────────────────────
import os
port = int(os.environ.get("PORT", 5000))
app.run(debug=False, host="0.0.0.0", port=port)
