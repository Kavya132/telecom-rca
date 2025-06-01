# 📡 Telecom Log Analyzer and Root Cause Prediction

This project is a web-based system that automates the analysis of telecom logs through the following key stages:

1. **Log Parsing and Insight Generation**
2. **Root Cause Analysis (RCA)**
3. **Reinforcement Learning for Continuous Improvement**

It provides an intuitive UI to input logs, view parsed insights, predict root causes, and enhance model learning with feedback.

---

## 🚀 Features

- 🧾 **Log Parsing**: Automatically extracts structured information from unstructured telecom logs.
- 🔍 **Insight Generation**: Highlights key entities, time stamps, error codes, and systems involved.
- 🧠 **Root Cause Analysis**: Uses a machine learning model to predict potential causes.
- ♻️ **Reinforcement Learning**: Allows users to give feedback to improve the model.
- 🎨 Responsive and visually appealing UI with CSS enhancements.

---

## 📂 Project Structure

telecom-rca/
│
├── static/
│ └── image.png # Background image used in UI
│
├── templates/
│ └── index.html and all html files# Frontend HTML (with embedded CSS & JS)
├──rca_training.py
├──rl.update.py
├──(all saved models with .pkl format)
├──insight_generation.ipynb
├──parse_logs.ipynb
├── app.py # Python backend (Flask-based)
├── requirements.txt # Python dependencies
└── README.md # Project documentation



---

## 🛠️ Technologies Used

- **Frontend**: HTML, CSS (custom design), JavaScript
- **Backend**: Python 3, Flask
- **ML**: (Simulated) ML model for RCA with mock predictions
- **Future**: Real-time parsing & model deployment

---


## 📊 Workflow Overview

### 1️⃣ Log Parsing and Insight Generation

- **Input**: Raw log text
- **Process**:
  - Extract error keywords
  - Parse timestamps, IPs, or node IDs
  - Identify log level (INFO, WARN, ERROR)
- **Output**: Parsed insights shown to user before prediction

### 2️⃣ Root Cause Analysis

- **Input**: Parsed or raw log
- **Output**: Top predicted cause + confidence %

### 3️⃣ Reinforcement Learning

- **Input**: Log + actual root cause
- **Process**: Update model with feedback and calculate reward
- **Output**: Confirmation with reward metric

---

## 🧪 Getting Started

### ✅ Prerequisites

- Python 3.x
- Flask (install via pip)

### 🔧 Installation

```bash
git clone https://github.com/your-username/telecom-rca.git
cd telecom-rca
pip install -r requirements.txt
python app.py
Visit: http://localhost:5000

💡 Example Use Cases
🔍 Insight Generation
Input: "2023-08-12 11:04:33 ERROR: Timeout at Node B"

Output:

Timestamp: 2023-08-12 11:04:33

Level: ERROR

Entity: Node B

🧠 Root Cause Prediction
Prediction: Network Congestion

Confidence: 87.43%

♻️ RL Feedback
Input Actual Cause: Hardware Fault

Reward: 8.24

📘 Future Enhancements
Plug in a real ML model

Save user logs and feedback

Graphical dashboard of RCA trends

NLP-based advanced log parsing
