# 🤖 Predictive Analytics & Probability Engine

A robust, real-time Machine Learning dashboard built with Python and Flask. This project is engineered to ingest sequential live API data, store historical records locally, and apply statistical models to predict incoming data patterns.

> **⚠️ IMPORTANT DISCLAIMER:**
> This project is an experimental Data Science application built strictly for **educational and awareness purposes**. The core objective is to demonstrate the mathematical limitations of Machine Learning models against pseudo-Random Number Generator (RNG) systems and server-side probability events. **We do not promote gambling or financial risk-taking.**

---

## 🚀 Features

This engine is equipped with advanced backend processing and a responsive analytical UI:

* **Live Data Ingestion:** Asynchronously fetches real-time sequence data from external REST APIs.
* **Continuous Machine Learning:** Utilizes Scikit-Learn's `SGDClassifier` with `.partial_fit()` to continuously train the model on new data without losing historical context.
* **Sliding Window Architecture:** Analyzes data using a dynamic "8-Window Size" pattern recognition logic to forecast subsequent outcomes.
* **Local Database Synchronization:** Automatically saves session data and synchronizes historical vectors using a lightweight local `SQLite` database (`local_game_history.db`).
* **Statistical Anomaly Detection (Variance Alert):** Automatically tracks loss streaks and triggers a high-variance UI warning if the system detects 5 consecutive negative outcomes (RNG death spiral).
* **Interactive Dashboard:** * Displays sequence trends, active session win rates, and global accuracy metrics.
  * Features a fully paginated historical data table directly fetched from the SQLite database.
  * Single-click CSV export for offline data analysis.
* **Dynamic Confidence Metric:** Calculates real-time prediction confidence based on mathematical variance and current streak multipliers.

---

## 🛠️ Technology Stack

* **Backend:** Python 3.x, Flask
* **Machine Learning:** Scikit-Learn (`SGDClassifier`), NumPy
* **Database:** SQLite3
* **API Handling:** Requests library
* **Frontend:** HTML5, CSS3, JavaScript (Jinja2 Templating)
