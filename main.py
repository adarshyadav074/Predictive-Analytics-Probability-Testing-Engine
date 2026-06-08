import os
import time
import pickle
import secrets
import threading
import requests
import sqlite3
import numpy as np
from flask import Flask, render_template, request, session, jsonify
from sklearn.linear_model import SGDClassifier
from threading import Lock

# Initialize Flask Application
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default-dev-secret-key-change-in-prod")

# ==========================================
# 🛑 USER CONFIGURATION 🛑
# ==========================================
DB_FILE = "local_game_history.db"
API_URL = os.environ.get("API_URL", "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json")
# ==========================================

MODEL_FILE_SIZE = "ai_model_size_v8.pkl"
MODEL_FILE_NUM = "ai_model_num_v8.pkl"
WINDOW_SIZE = 8

state_lock = Lock()

session_history = []
game_state = {
    "next_issue": "Loading...",
    "prediction": "Wait...",
    "sure_numbers": "Wait...",
    "confidence": "50.0%",
    "ai_status": "Initializing System...",
    "max_win_streak": 0,
    "win_rate": 0.0,
    "history_table": []
}

# --- Database Management (SQLite) ---
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        with get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS game_history (
                    issue TEXT PRIMARY KEY,
                    prediction_text TEXT,
                    result_text TEXT,
                    color TEXT,
                    number INTEGER
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_issue ON game_history(issue DESC);')
            conn.commit()
    except Exception as e:
        print(f"[WARNING] Database Initialization Error: {e}")

def save_to_db(issue, prediction_text, result_text, color, number):
    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT OR IGNORE INTO game_history (issue, prediction_text, result_text, color, number)
                VALUES (?, ?, ?, ?, ?)
            ''', (issue, prediction_text, result_text, color, number))
            conn.commit()
    except Exception as e:
        print(f"[WARNING] save_to_db failed: {e}")

def calculate_stats():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT result_text FROM game_history 
                WHERE prediction_text NOT LIKE '%VISION_AI%' 
                AND prediction_text NOT LIKE 'Wait%'
                AND prediction_text != ''
                AND result_text IS NOT NULL 
                AND result_text != ''
                ORDER BY issue ASC
            ''')
            rows = cursor.fetchall()

        max_win_streak = 0
        current_win_streak = 0
        total_wins = 0

        for row in rows:
            res = row[0]
            if "WIN" in res or "JACKPOT" in res:
                current_win_streak += 1
                total_wins += 1
                if current_win_streak > max_win_streak: 
                    max_win_streak = current_win_streak
            elif "LOSS" in res:
                current_win_streak = 0
        
        total_games = len(rows)
        win_rate = round((total_wins / total_games) * 100, 2) if total_games > 0 else 0.0
        return max_win_streak, win_rate
    except Exception as e:
        print(f"[WARNING] calculate_stats failed: {e}")
        return 0, 0.0

# --- Machine Learning Operations ---
def load_or_create_models():
    if os.path.exists(MODEL_FILE_SIZE) and os.path.exists(MODEL_FILE_NUM):
        try:
            with open(MODEL_FILE_SIZE, 'rb') as f1, open(MODEL_FILE_NUM, 'rb') as f2:
                print("Models loaded successfully from disk.")
                return pickle.load(f1), pickle.load(f2)
        except Exception as e:
            print(f"[WARNING] Failed to load models: {e}")
            
    print("Initializing fresh SGD Classifiers...")
    return SGDClassifier(loss='log_loss', random_state=42), SGDClassifier(loss='log_loss', random_state=24)

def save_models(model_size, model_num):
    try:
        with open(MODEL_FILE_SIZE, 'wb') as f1, open(MODEL_FILE_NUM, 'wb') as f2:
            pickle.dump(model_size, f1)
            pickle.dump(model_num, f2)
    except Exception as e:
        print(f"[WARNING] Failed to save models: {e}")

def sync_model_with_db():
    global ml_model_size, ml_model_num
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT number FROM game_history WHERE number IS NOT NULL ORDER BY issue ASC")
            rows = cursor.fetchall()

        if len(rows) < WINDOW_SIZE + 2: 
            return
            
        num_history = [r[0] for r in rows]
        size_history = [1 if n >= 5 else 0 for n in num_history]

        X_size, y_size, X_num, y_num = [], [], [], []
        for i in range(len(size_history) - WINDOW_SIZE):
            X_size.append(size_history[i : i + WINDOW_SIZE])
            y_size.append(size_history[i + WINDOW_SIZE])
            X_num.append(num_history[i : i + WINDOW_SIZE])
            y_num.append(num_history[i + WINDOW_SIZE])

        with state_lock:
            ml_model_size.partial_fit(X_size, y_size, classes=np.array([0, 1]))
            ml_model_num.partial_fit(X_num, y_num, classes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        
        save_models(ml_model_size, ml_model_num)
        print(f"Model synchronization complete. Evaluated {len(rows)} historical records.")
    except Exception as e:
        print(f"[WARNING] Database sync failed: {e}")

init_db()
ml_model_size, ml_model_num = load_or_create_models()
sync_model_with_db()

initial_max_win, initial_win_rate = calculate_stats()
with state_lock:
    game_state.update({
        "max_win_streak": initial_max_win,
        "win_rate": initial_win_rate
    })

def process_ai(results_list, train=True):
    global ml_model_size, ml_model_num
    
    if len(results_list) < WINDOW_SIZE + 2:
        return "WAIT", [0, 0], f"Buffering {WINDOW_SIZE} sequences...", "50.0%"

    size_history = [1 if int(item["number"]) >= 5 else 0 for item in results_list]
    num_history = [int(item["number"]) for item in results_list]
    size_history.reverse()
    num_history.reverse()

    if train:
        X_size = [size_history[-WINDOW_SIZE-1:-1]]
        y_size = [size_history[-1]]
        X_num = [num_history[-WINDOW_SIZE-1:-1]]
        y_num = [num_history[-1]]

        with state_lock:
            ml_model_size.partial_fit(X_size, y_size)
            ml_model_num.partial_fit(X_num, y_num)
        save_models(ml_model_size, ml_model_num)

    latest_size_pattern = [size_history[-WINDOW_SIZE:]]
    latest_num_pattern = [num_history[-WINDOW_SIZE:]]

    with state_lock:
        prediction_size_num = ml_model_size.predict(latest_size_pattern)[0]
        size_probs = ml_model_size.predict_proba(latest_size_pattern)[0]
        num_probabilities = ml_model_num.predict_proba(latest_num_pattern)[0]

    predicted_size = "BIG" if prediction_size_num == 1 else "SMALL"
    
    raw_conf = max(size_probs)
    confidence_val = round(raw_conf * 100, 1)

    valid_nums = [5, 6, 7, 8, 9] if predicted_size == "BIG" else [0, 1, 2, 3, 4]
    valid_probs = {num: num_probabilities[num] for num in valid_nums}
    sure_numbers = sorted(valid_probs, key=valid_probs.get, reverse=True)[:2]
    sure_numbers.sort()

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM game_history')
            total_saved = cursor.fetchone()[0]
        ai_status = f"Local DB Records: {total_saved}"
    except Exception:
        ai_status = "Local DB: Syncing..."

    return predicted_size, sure_numbers, ai_status, f"{confidence_val}%"

def run_background_worker():
    global game_state, session_history
    last_server_issue = None
    current_prediction = None
    retry_delay = 2 

    while True:
        try:
            current_ts = int(time.time() * 1000)
            response = requests.get(f"{API_URL}?ts={current_ts}", timeout=10)
            results = response.json().get("data", {}).get("list", [])

            if not results:
                time.sleep(retry_delay)
                continue

            retry_delay = 2
            latest = results[0]
            server_issue = latest["issueNumber"]

            if server_issue != last_server_issue:
                if last_server_issue is not None:
                    actual_num = int(latest["number"])
                    actual_size = "BIG" if actual_num >= 5 else "SMALL"
                    
                    if current_prediction:
                        predicted_size = current_prediction["prediction"]
                        sure_nums = current_prediction["sure_numbers"]

                        size_match = actual_size == predicted_size
                        number_match = actual_num in sure_nums

                        if size_match and number_match: 
                            status, color = "🎉 JACKPOT", "#ffeb3b"
                        elif size_match or number_match: 
                            status, color = "✅ WIN", "#4caf50"
                        else: 
                            status, color = "❌ LOSS", "#f44336"

                        result_text = f"{status} ({actual_size} {actual_num})"
                        prediction_text = f"{predicted_size} ({sure_nums[0]}-{sure_nums[1]})"

                        save_to_db(current_prediction["issue"], prediction_text, result_text, color, actual_num)
                        
                        with state_lock:
                            session_history.insert(0, {
                                "issue": current_prediction["issue"],
                                "prediction_text": prediction_text,
                                "result_text": result_text,
                                "color": color,
                                "number": actual_num
                            })
                            game_state["history_table"] = session_history
                        
                        w_streak, w_rate = calculate_stats()
                        with state_lock:
                            game_state.update({
                                "max_win_streak": w_streak,
                                "win_rate": w_rate
                            })

                    predicted_size, sure_numbers, ai_status, conf = process_ai(results, train=True)
                else:
                    predicted_size, sure_numbers, ai_status, conf = process_ai(results, train=False)
                
                if predicted_size != "WAIT":
                    next_issue = str(int(server_issue) + 1)
                    current_prediction = {
                        "issue": next_issue, 
                        "prediction": predicted_size, 
                        "sure_numbers": sure_numbers
                    }

                    with state_lock:
                        game_state.update({
                            "next_issue": next_issue,
                            "prediction": predicted_size,
                            "sure_numbers": f"{sure_numbers[0]} - {sure_numbers[1]}",
                            "confidence": conf,
                            "ai_status": ai_status
                        })
                    
                last_server_issue = server_issue

            time.sleep(2)
        except Exception as e:
            # ✅ FIX 6: Exponential Backoff Applied
            print(f"[WARNING] Worker API Error: {e}")
            time.sleep(retry_delay)
            retry_delay = min(60, retry_delay * 2)

# --- FLASK ROUTES ---

@app.route("/api/state")
def api_state():
    page = request.args.get('page', 1, type=int)

    with state_lock:
        current_state = game_state.copy()
        current_history = list(session_history)

    start_issue = session.get('start_issue', '0')
    user_history = []
    
    if start_issue.isdigit():
        start_issue_int = int(start_issue)
        for row in current_history:
            if row["issue"].isdigit() and int(row["issue"]) >= start_issue_int:
                user_history.append(row)
                
    user_wins = sum(1 for row in user_history if "WIN" in row["result_text"] or "JACKPOT" in row["result_text"])
    user_win_rate = round((user_wins / len(user_history)) * 100, 2) if user_history else 0.0
    
    current_loss_streak = 0
    for row in user_history:
        if "LOSS" in row["result_text"]: current_loss_streak += 1
        else: break

    trend_dots = []
    for row in user_history[:10]:
        num = row.get("number")
        if num is not None:
            if num in [1, 3, 7, 9]: trend_dots.append("🟢")
            elif num in [2, 4, 6, 8]: trend_dots.append("🔴")
            elif num in [0, 5]: trend_dots.append("🟣")
    trend_dots.reverse()
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM game_history WHERE result_text IS NOT NULL AND result_text != ''")
            total_records = cursor.fetchone()[0]
            
            offset = (page - 1) * 10
            cursor.execute('''
                SELECT issue, prediction_text, result_text, color 
                FROM game_history 
                WHERE result_text IS NOT NULL AND result_text != ''
                ORDER BY issue DESC 
                LIMIT 10 OFFSET ?
            ''', (offset,))
            db_rows = cursor.fetchall()
        
        paginated_history = [{"issue": r[0], "prediction_text": r[1], "result_text": r[2], "color": r[3]} for r in db_rows]
        total_pages = max(1, (total_records + 9) // 10)
    except Exception as e:
        print(f"[WARNING] DB Pagination Error: {e}")
        paginated_history = []
        total_pages = 1

    return jsonify({
        "next_issue": current_state["next_issue"],
        "prediction": current_state["prediction"],
        "sure_numbers": current_state["sure_numbers"],
        "confidence": current_state["confidence"],
        "ai_status": current_state["ai_status"],
        "max_win_streak": current_state["max_win_streak"],
        "win_rate": current_state["win_rate"],
        "current_win_rate": user_win_rate,
        "current_loss_streak": current_loss_streak,
        "trend": " ".join(trend_dots) if trend_dots else "Awaiting Data...",
        "paginated_history": paginated_history,
        "current_page": page,
        "total_pages": total_pages
    })

@app.route("/")
def index():
    if 'visitor_logged' not in session:
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
        if ip_address: ip_address = ip_address.split(',')[0].strip()
        print(f"[Session Started] IP: {ip_address} | OS: {request.user_agent.string}")
        session['visitor_logged'] = True

    now = time.time()
    last_seen = session.get('last_seen', 0)
    
    with state_lock:
        current_issue = game_state.get('next_issue', '0')
        
    if (now - last_seen > 15) or ('start_issue' not in session):
        session['start_issue'] = current_issue
        session.pop('visitor_logged', None) 
        
    session['last_seen'] = now
    
    return render_template("index.html")

if __name__ == "__main__":
    threading.Thread(target=run_background_worker, daemon=True).start()
    print("🚀 Predictive Analytics Engine starting on Localhost (127.0.0.1:5000)...")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)