---
title: "OpenEnv Demo - Email, Cleaner, Scheduler"
emoji: 📬🧹📅
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
license: mit
---

# Triaging, Cleaning, and Scheduling Environments

This repository contains three **OpenEnv‑compliant environments** designed for reinforcement learning agents:

- **Email Triage** → classify emails into categories (urgent, personal, spam, informational).
- **Data Cleaning** → correct noisy data entries by proposing cleaned values.
- **Scheduling** → extract structured details (date, time, attendees, location) from natural language requests.

Each environment implements the standard `reset()`, `step()`, and `state()` methods and uses typed dataclasses for actions, observations, and states.

---

## 📌 Action & Observation Spaces

### Email
- **EmailAction**
  - `category: Literal["urgent","personal","spam","informational"]`
- **EmailObservation**
  - `subject: str`
  - `body: str`
  - `sender: str`
  - `timestamp: str`
  - `reward: float`
  - `done: bool`
  - `info: Dict[str, Any]`
- **EmailState**
  - `episode_id: str`
  - `step_count: int`
  - `done: bool`

### Cleaner
- **CleanerAction**
  - `field: str`
  - `new_value: str`
- **CleanerObservation**
  - `current_values: Dict[str, str]`
  - `reward: float`
  - `done: bool`
  - `info: Dict[str, Any]`
- **CleanerState**
  - `episode_id: str`
  - `step_count: int`
  - `done: bool`

### Scheduler
- **SchedulerAction**
  - `field: str`
  - `new_value: Any`
- **SchedulerObservation**
  - `raw_string: str`
  - `current_values: Dict[str, Any]`
  - `reward: float`
  - `done: bool`
  - `info: Dict[str, Any]`
- **SchedulerState**
  - `episode_id: str`
  - `step_count: int`
  - `done: bool`

---

## 🎯 Task Descriptions

- **Email Triage (easy)**  
  Classify emails into one of four categories.

- **Data Cleaning (medium)**  
  Correct noisy entries by proposing cleaned values for fields.

- **Scheduling (hard)**  
  Extract structured details (date, time, attendees, location) from natural language scheduling requests.

---

## 📊 Baseline Scores

We provide a `baseline.py` script that runs each environment with a **perfect agent** (always applies the ground truth). Example output:

[Email] reward=1.0, done=True
[Cleaner] Step 1: reward=1.0, done=True
[Scheduler] Step 4: reward=1.0, done=True

Baseline average scores over 5 episodes:
email_triage: 1.00
data_cleaning: 1.00
scheduling: 1.00

Code

- These scores represent an upper bound (perfect agent).  
- Random agents would score lower (e.g., ~0.25 for Email Triage).

---

## ⚙️ Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
Run baseline:

bash
python baseline.py
Validate OpenEnv compliance:

bash
openenv validate
🚀 Deployment Notes
Includes a Dockerfile for containerized deployment.

Compatible with Hugging Face Spaces using FastAPI entrypoints.

Default entrypoint runs on port 7860.

📂 File Structure
Code
.
├── core/
│   └── model.py          # Dataclass definitions
├── envs/
│   ├── email_triage.py   # Email environment
│   ├── data_cleaner.py   # Cleaner environment
│   └── scheduler.py      # Scheduler environment
├── baseline.py           # Baseline inference script
├── openenv.yaml          # OpenEnv specification
├── Dockerfile            # Containerization
└── README.md             # Documentation
✅ Compliance
This repository follows the OpenEnv specification:

Typed Action/Observation/State dataclasses.

reset(), step(), state() methods.

openenv.yaml with schemas and tasks.

Baseline scores included in README.

Containerized deployment support.