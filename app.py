from fastapi import FastAPI
from envs.email_triage import EmailTriageEnv
from envs.data_cleaner import DataCleanerEnv
from envs.scheduler import SchedulerEnv
from fastapi.responses import RedirectResponse
from core.model import EmailAction, CleanerAction, SchedulerAction



# Example datasets
email_dataset = [
    {"subject": "Meeting tomorrow", "body": "Don't forget the 10 AM meeting.",
     "sender": "boss@example.com", "timestamp": "2026-04-03T09:00:00", "label": "urgent"},
    {"subject": "Win a free iPhone!", "body": "Click here to claim your prize.",
     "sender": "scam@example.com", "timestamp": "2026-04-02T14:30:00", "label": "spam"}
]

cleaning_dataset = [
    {"raw_entry": {"name": "john DOE", "phone": "123-456-7890"},
     "ground_truth": {"name": "John Doe", "phone": "1234567890"}}
]

scheduling_dataset = [
    {"raw_request": "Meet John tomorrow afternoon in conference room",
     "ground_truth": {"date": "2026-04-06", "time": "15:00",
                      "attendees": ["John"], "location": "conference room"}}
]

# Initialize environments
email_env = EmailTriageEnv(email_dataset)
cleaning_env = DataCleanerEnv(cleaning_dataset)
scheduling_env = SchedulerEnv(scheduling_dataset)

app = FastAPI()

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

# Email endpoints
@app.post("/email/reset")
def reset_email():
    return email_env.reset()

@app.post("/email/step")
def step_email(action: EmailAction):
    return email_env.step(action)

# Cleaning endpoints
@app.post("/cleaning/reset")
def reset_cleaning():
    return cleaning_env.reset()

@app.post("/cleaning/step")
def step_cleaning(action: CleanerAction):
    return cleaning_env.step(action)

# Scheduling endpoints
@app.post("/scheduling/reset")
def reset_scheduling():
    return scheduling_env.reset()

@app.post("/scheduling/step")
def step_scheduling(action: SchedulerAction):
    return scheduling_env.step(action)
