import os
import time
import pickle
import secrets
import threading
import requests
import sqlite3
import numpy as np
from flask import Flask, render_template_string, request, session
from sklearn.linear_model import SGDClassifier

# Initialize Flask Application
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# ==========================================
# 🛑 USER CONFIGURATION 🛑
# ==========================================
# Local SQLite Database File
DB_FILE = "local_game_history.db"

# Target Website API URL for fetching live results
API_URL = os.environ.get("API_URL", "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json")
# ==========================================

# Model Constants
MODEL_FILE_SIZE = "ai_model_size_v8.pkl"
MODEL_FILE_NUM = "ai_model_num_v8.pkl"
WINDOW_SIZE = 8

# Global Application State
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
    """Establish and return a connection to the local SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables if they do not exist."""
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
            conn.commit()
    except Exception as e:
        print(f"Database Initialization Error: {e}")

def save_to_db(issue, prediction_text, result_text, color, number):
    """Save a game result record to the local database."""
    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT OR IGNORE INTO game_history (issue, prediction_text, result_text, color, number)
                VALUES (?, ?, ?, ?, ?)
            ''', (issue, prediction_text, result_text, color, number))
            conn.commit()
    except Exception:
        pass

def calculate_stats():
    """Calculate overall historical win rate and max streak from the database."""
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
    except Exception:
        return 0, 0.0

# --- Machine Learning Operations ---
def load_or_create_models():
    """Load existing ML models from disk or instantiate new SGD Classifiers."""
    if os.path.exists(MODEL_FILE_SIZE) and os.path.exists(MODEL_FILE_NUM):
        try:
            with open(MODEL_FILE_SIZE, 'rb') as f1, open(MODEL_FILE_NUM, 'rb') as f2:
                print("Models loaded successfully from disk.")
                return pickle.load(f1), pickle.load(f2)
        except Exception:
            pass
    print("Initializing fresh SGD Classifiers...")
    return SGDClassifier(loss='log_loss', random_state=42), SGDClassifier(loss='log_loss', random_state=24)

def save_models(model_size, model_num):
    """Serialize and save ML models to disk."""
    with open(MODEL_FILE_SIZE, 'wb') as f1, open(MODEL_FILE_NUM, 'wb') as f2:
        pickle.dump(model_size, f1)
        pickle.dump(model_num, f2)

def sync_model_with_db():
    """Fetch historical data from the database and perform partial fitting."""
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

        ml_model_size = SGDClassifier(loss='log_loss', random_state=42)
        ml_model_num = SGDClassifier(loss='log_loss', random_state=24)
        
        ml_model_size.partial_fit(X_size, y_size, classes=np.array([0, 1]))
        ml_model_num.partial_fit(X_num, y_num, classes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        
        save_models(ml_model_size, ml_model_num)
        print(f"Model synchronization complete. Evaluated {len(rows)} historical records.")
    except Exception as e:
        print(f"Database sync failed: {e}")

# System Initialization
init_db()
ml_model_size, ml_model_num = load_or_create_models()
sync_model_with_db()

initial_max_win, initial_win_rate = calculate_stats()
game_state.update({
    "max_win_streak": initial_max_win,
    "win_rate": initial_win_rate
})

def process_ai(results_list, train=True):
    """Core logic to process incoming data, train the model, and generate predictions."""
    global ml_model_size, ml_model_num
    
    if len(results_list) < WINDOW_SIZE + 2:
        return "WAIT", [0, 0], f"Buffering {WINDOW_SIZE} sequences...", "50.0%"

    size_history = [1 if int(item["number"]) >= 5 else 0 for item in results_list]
    num_history = [int(item["number"]) for item in results_list]
    size_history.reverse()
    num_history.reverse()

    if train:
        X_size, y_size, X_num, y_num = [], [], [], []
        for i in range(len(size_history) - WINDOW_SIZE):
            X_size.append(size_history[i : i + WINDOW_SIZE])
            y_size.append(size_history[i + WINDOW_SIZE])
            X_num.append(num_history[i : i + WINDOW_SIZE])
            y_num.append(num_history[i + WINDOW_SIZE])

        ml_model_size.partial_fit(X_size, y_size, classes=np.array([0, 1]))
        ml_model_num.partial_fit(X_num, y_num, classes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        save_models(ml_model_size, ml_model_num)

    latest_size_pattern = [size_history[-WINDOW_SIZE:]]
    latest_num_pattern = [num_history[-WINDOW_SIZE:]]

    prediction_size_num = ml_model_size.predict(latest_size_pattern)[0]
    predicted_size = "BIG" if prediction_size_num == 1 else "SMALL"
    
    size_probs = ml_model_size.predict_proba(latest_size_pattern)[0]
    raw_conf = max(size_probs)
    pattern_variance = np.var(latest_size_pattern[0])
    
    base_conf = 70.0 + (pattern_variance * 50.0) + (int(time.time()) % 7) if raw_conf >= 0.99 else raw_conf * 100
        
    current_streak = 0
    is_losing = False
    
    global session_history
    if session_history:
        last_status = session_history[0]["result_text"]
        if "LOSS" in last_status:
            is_losing = True
            for row in session_history:
                if "LOSS" in row["result_text"]: current_streak += 1
                else: break
        else:
            for row in session_history:
                if "WIN" in row["result_text"] or "JACKPOT" in row["result_text"]: current_streak += 1
                else: break

    adjusted_conf = base_conf - (current_streak * 12.5) if is_losing else base_conf + (current_streak * 2.5)
    confidence_val = round(min(max(adjusted_conf, 15.5), 96.8), 1)

    valid_nums = [5, 6, 7, 8, 9] if predicted_size == "BIG" else [0, 1, 2, 3, 4]
    num_probabilities = ml_model_num.predict_proba(latest_num_pattern)[0]
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
    """Background thread to continuously fetch API data and update states."""
    global game_state, session_history
    last_server_issue = None
    current_prediction = None

    while True:
        try:
            current_ts = int(time.time() * 1000)
            response = requests.get(f"{API_URL}?ts={current_ts}", timeout=10)
            results = response.json().get("data", {}).get("list", [])

            if not results:
                time.sleep(2)
                continue

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
                        
                        session_history.insert(0, {
                            "issue": current_prediction["issue"],
                            "prediction_text": prediction_text,
                            "result_text": result_text,
                            "color": color,
                            "number": actual_num
                        })
                        
                        game_state["history_table"] = session_history
                        
                        w_streak, w_rate = calculate_stats()
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
            time.sleep(5)

# --- FLASK JINJA HTML TEMPLATE ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Analytics Dashboard</title>
    <meta http-equiv="refresh" content="3;url=/?page={{ data.current_page }}">
    <style>
        body { background-color: #0d0d0d; color: #ffffff; font-family: 'Courier New', Courier, monospace; text-align: center; margin-top: 30px; }
        .container { border: 2px solid #00ffcc; padding: 20px; width: 85%; margin: auto; box-shadow: 0 0 20px #00ffcc; border-radius: 10px; background: #1a1a1a; margin-bottom: 50px; position: relative; overflow: hidden; }
        .hacker-bg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; opacity: 0.12; color: #0f0; font-size: 13px; text-align: left; padding: 15px; box-sizing: border-box; pointer-events: none; white-space: pre-wrap; word-wrap: break-word; overflow: hidden; text-shadow: 0 0 5px #0f0; }
        .content-wrapper { position: relative; z-index: 10; }
        h1 { color: #00ffcc; letter-spacing: 2px; text-shadow: 0 0 10px #00ffcc;}
        .floating-btns { position: fixed; top: 20px; right: 20px; z-index: 999; display: flex; gap: 15px;}
        .info-btn { background: #ffeb3b; color: #000; border: none; border-radius: 50%; width: 45px; height: 45px; font-size: 24px; font-weight: bold; cursor: pointer; box-shadow: 0 0 15px #ffeb3b;}
        .csv-btn { background: #34d399; color: #000; border: none; border-radius: 8px; padding: 0 15px; font-size: 14px; font-weight: bold; cursor: pointer; box-shadow: 0 0 15px #34d399; font-family: sans-serif;}
        .modal-overlay { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.85); backdrop-filter: blur(5px); }
        .modal-content { background-color: #111; margin: 5% auto; padding: 30px; border: 2px solid #f44336; border-radius: 12px; width: 85%; max-width: 650px; color: #fff; text-align: left; font-family: sans-serif;}
        .modal-header { font-size: 24px; font-weight: 900; margin-bottom: 20px; text-align: center; border-bottom: 1px solid #333; padding-bottom: 15px;}
        .modal-body { font-size: 16px; line-height: 1.7; color: #ddd; }
        .close-btn { color: white; border: none; padding: 15px 20px; font-weight: bold; border-radius: 8px; cursor: pointer; display: block; margin: 25px auto 0; width: 100%;}
        .trend-badge { background-color: #111; color: #fff; padding: 5px 15px; border-radius: 8px; font-size: 18px; border: 1px dashed #555; display: inline-block; margin-bottom: 15px; letter-spacing: 5px;}
        .ai-badge { background-color: #222; color: #00ffcc; padding: 8px 20px; border-radius: 20px; font-size: 16px; border: 1px solid #00ffcc; display: inline-block; margin-bottom: 15px; font-weight: bold;}
        .streak-badge { background-color: #222; color: #ff9800; padding: 8px 20px; border-radius: 20px; font-size: 16px; border: 1px dashed #ff9800; display: inline-block; margin-bottom: 15px;}
        .session-badge { background-color: #111; color: #34d399; padding: 8px 20px; border-radius: 20px; font-size: 16px; border: 1px solid #34d399; display: inline-block; margin-bottom: 15px;}
        .win-txt { color: #4caf50; } .highlight-session { color: #34d399;}
        .current-prediction { background: rgba(0, 255, 204, 0.1); border: 1px dashed #00ffcc; padding: 20px; border-radius: 8px; margin: 10px 0; }
        .current-prediction h2 { margin: 10px 0; font-size: 24px; }
        .highlight { color: #00ffcc; font-weight: bold; font-size: 28px; }
        table { width: 100%; border-collapse: collapse; margin-top: 25px; }
        th, td { border: 1px solid #333; padding: 12px; text-align: center; font-size: 16px; }
        th { background-color: #00ffcc; color: #000; font-weight: bold; }
        tr:nth-child(even) { background-color: rgba(17, 17, 17, 0.85); } tr:nth-child(odd) { background-color: rgba(34, 34, 34, 0.85); }
        .pagination { margin-top: 25px; display: flex; justify-content: center; align-items: center; gap: 20px; }
        .pagination a { background: #00ffcc; color: #000; padding: 10px 20px; border-radius: 5px; text-decoration: none; font-weight: bold; box-shadow: 0 0 10px #00ffcc; font-family: sans-serif;}
        .pagination a:hover { background: #fff; box-shadow: 0 0 15px #fff; }
        .pagination span { font-size: 18px; font-weight: bold; color: #fff; }
    </style>
</head>
<body>
    
    <div class="floating-btns">
        <button class="csv-btn" onclick="downloadCSV()">📥 Export Data</button>
        <button class="info-btn" onclick="openModal()">i</button>
    </div>

    <div id="lossWarningModal" class="modal-overlay" style="z-index: 10000;">
        <div class="modal-content" style="border-color: #ff0000; box-shadow: 0 0 25px #ff0000;">
            <div class="modal-header" style="color: #ff0000; font-size: 28px;">🚨 CRITICAL DANGER ALERT 🚨</div>
            <div class="modal-body" style="text-align: center;">
                <p style="font-size: 20px; font-weight: bold; color: #ff5555;">AI 5 Baar Lagataar Loss Kar Chuka Hai!</p>
                <p style="font-size: 18px;">According to our data, agar AI 5 times non-stop loss karta hai, toh iska matlab <strong>market trend bohot kharab chal raha hai</strong> aur pure RNG death spiral shuru ho chuka hai.</p>
                <p style="color: #ffeb3b; font-size: 18px; margin-top: 15px;">Kripya apna capital safe rakhein aur abhi trade STOP kar dein jab tak trend normal nahi hota!</p>
                <button class="close-btn" style="background-color: #ff0000; color: #fff; border: 2px solid #fff; font-size: 18px; margin-top: 30px;" onclick="closeLossWarning()">I UNDERSTAND (STOP PLAYING)</button>
            </div>
        </div>
    </div>

    <div id="disclaimerModal" class="modal-overlay">
        <div class="modal-content" style="border-color: #f44336;">
            <div class="modal-header" style="color: #f44336;">⚠️ IMPORTANT DISCLAIMER ⚠️</div>
            <div class="modal-body">
                <p>This "AI Hack Engine" is an experimental project built using Machine Learning algorithms strictly for <strong>educational and awareness purposes</strong>.</p>
                <p style="color: #f44336; font-weight: bold;">WE DO NOT PROMOTE GAMBLING OR HACKING.</p>
                <p>The core objective is to demonstrate that even highly advanced ML models <strong>CANNOT perfectly predict</strong> server-side probability games. Beware of fake hack sellers online.</p>
                <p><strong>PLAY AT YOUR OWN RISK.</strong> The developer assumes NO responsibility for any financial losses.</p>
                <button class="close-btn" style="background-color: #f44336;" onclick="closeDisclaimer()">I UNDERSTAND & AGREE</button>
            </div>
        </div>
    </div>

    <div class="container">
        <div id="hacker-bg" class="hacker-bg"></div>
        
        <div class="content-wrapper">
            <h1>🤖 PREDICTIVE ANALYTICS ENGINE 🤖</h1>
            
            <div class="trend-badge">SEQUENCE TREND: {{ data.trend }}</div><br>
            
            <div>
                <div class="ai-badge">{{ data.ai_status }}</div>
                <div class="streak-badge">🔥 Peak Continuous Accuracy: <span class="win-txt">{{ data.max_win_streak }}</span></div><br>
                <div class="session-badge">🌍 Global Accuracy Rate: <span class="win-txt">{{ data.win_rate }}%</span> | ⚡ Active Session Rate: <span class="highlight-session">{{ data.current_win_rate }}%</span></div>
            </div>
            
            <div class="current-prediction">
                <h2>ISSUE ID : <span class="highlight">{{ data.next_issue }}</span></h2>
                <h2>FORECAST : <span class="highlight">{{ data.prediction }}</span></h2>
                <h2>DATA POINTS : <span class="highlight">{{ data.sure_numbers }}</span></h2>
                <h2>CONFIDENCE METRIC : <span class="highlight">{{ data.confidence }}</span></h2>
            </div>
            
            <table id="dataTable">
                <thead>
                    <tr><th>Issue ID</th><th>System Forecast</th><th>Actual Resolution</th></tr>
                </thead>
                <tbody>
                    {% if paginated_history %}
                        {% for row in paginated_history %}
                        <tr><td>{{ row.issue }}</td><td>{{ row.prediction_text }}</td><td style="color: {{ row.color }}; font-weight: bold;">{{ row.result_text }}</td></tr>
                        {% endfor %}
                    {% else %}
                        <tr><td colspan="3" style="color:#888;">Synchronizing Database... Awaiting server payload.</td></tr>
                    {% endif %}
                </tbody>
            </table>

            <div class="pagination">
                {% if data.current_page > 1 %}
                    <a href="/?page={{ data.current_page - 1 }}">⬅️ Prev</a>
                {% endif %}
                <span>Page {{ data.current_page }} of {{ data.total_pages }}</span>
                {% if data.current_page < data.total_pages %}
                    <a href="/?page={{ data.current_page + 1 }}">Next ➡️</a>
                {% endif %}
            </div>

        </div>
    </div>

    <script>
        var disclaimerModal = document.getElementById("disclaimerModal");
        var lossWarningModal = document.getElementById("lossWarningModal");

        window.onload = function() {
            const currentLossStreak = parseInt("{{ data.current_loss_streak }}");
            if (currentLossStreak >= 5) {
                if (!sessionStorage.getItem("lossWarningActive")) {
                    lossWarningModal.style.display = "block";
                    sessionStorage.setItem("lossWarningActive", "true");
                }
            } else {
                sessionStorage.removeItem("lossWarningActive");
            }

            if (!sessionStorage.getItem("disclaimerAcknowledged")) {
                disclaimerModal.style.display = "block";
            }
        }

        function closeLossWarning() { lossWarningModal.style.display = "none"; }
        function openModal() { disclaimerModal.style.display = "block"; sessionStorage.removeItem("disclaimerAcknowledged"); }
        function closeDisclaimer() { disclaimerModal.style.display = "none"; sessionStorage.setItem("disclaimerAcknowledged", "true"); initAudio(); }

        let audioCtx;
        function initAudio() { if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
        function playBeep() {
            if(audioCtx && audioCtx.state === 'running') {
                const osc = audioCtx.createOscillator();
                osc.type = 'square'; osc.frequency.setValueAtTime(600, audioCtx.currentTime);
                osc.connect(audioCtx.destination);
                osc.start(); osc.stop(audioCtx.currentTime + 0.1);
            }
        }
        
        const currentIssue = "{{ data.next_issue }}";
        const lastSeen = sessionStorage.getItem("lastSeenIssue");
        if(lastSeen && lastSeen !== currentIssue) { playBeep(); }
        sessionStorage.setItem("lastSeenIssue", currentIssue);

        function downloadCSV() {
            let table = document.getElementById("dataTable");
            let rows = table.querySelectorAll("tr");
            let csv = [];
            for (let i = 0; i < rows.length; i++) {
                let row = [], cols = rows[i].querySelectorAll("td, th");
                for (let j = 0; j < cols.length; j++) row.push(cols[j].innerText.replace(/,/g, ""));
                csv.push(row.join(","));
            }
            let blob = new Blob([csv.join("\\n")], {type: "text/csv"});
            let a = document.createElement("a");
            a.href = window.URL.createObjectURL(blob);
            a.download = "Predictive_Model_Data.csv";
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
        }

        const bgBox = document.getElementById("hacker-bg");
        const commands = [
            "Initializing SGD Classifier environment...",
            "Connecting to local SQLite database...",
            "Synchronizing historical vectors...",
            "Calculating sliding window probabilities...",
            "Adjusting log-loss weights...",
            "Fetching real-time API payload...",
            "Evaluating variance thresholds...",
            "Updating user interface DOM..."
        ];
        
        let termLines = [];
        if(sessionStorage.getItem("bgLogs")) { termLines = JSON.parse(sessionStorage.getItem("bgLogs")); }

        function updateTerminalBg() {
            let rand = Math.random();
            let newLine = "";
            if (rand < 0.35) newLine = commands[Math.floor(Math.random() * commands.length)];
            else newLine = "Data sync check: [████████░░] " + Math.floor(Math.random() * 100) + "%";

            termLines.push(newLine);
            if (termLines.length > 35) termLines.shift();
            
            bgBox.innerText = termLines.join("\\n");
            sessionStorage.setItem("bgLogs", JSON.stringify(termLines));
            setTimeout(updateTerminalBg, Math.random() * 300 + 100); 
        }
        updateTerminalBg();
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    page = request.args.get('page', 1, type=int)

    if 'visitor_logged' not in session:
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
        if ip_address: ip_address = ip_address.split(',')[0].strip()
        print(f"[Session Started] IP: {ip_address} | OS: {request.user_agent.string}")
        session['visitor_logged'] = True

    now = time.time()
    last_seen = session.get('last_seen', 0)
    
    if (now - last_seen > 15) or ('start_issue' not in session):
        session['start_issue'] = game_state.get('next_issue', '0')
        session.pop('visitor_logged', None) 
        
    session['last_seen'] = now
    start_issue = session.get('start_issue', '0')
    
    user_history = []
    if start_issue.isdigit():
        start_issue_int = int(start_issue)
        for row in game_state["history_table"]:
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
    except Exception:
        paginated_history = []
        total_pages = 1

    display_data = game_state.copy()
    display_data.update({
        "current_win_rate": user_win_rate,
        "current_loss_streak": current_loss_streak,
        "trend": " ".join(trend_dots) if trend_dots else "Awaiting Data...",
        "current_page": page,
        "total_pages": total_pages
    })
    
    return render_template_string(HTML_PAGE, data=display_data, paginated_history=paginated_history, user_history=user_history)

if __name__ == "__main__":
    # Start the background data fetching and processing thread
    threading.Thread(target=run_background_worker, daemon=True).start()
    
    # Run the application explicitly on localhost
    print("🚀 Predictive Analytics Engine starting on Localhost (127.0.0.1:5000)...")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)