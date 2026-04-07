from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from envs.email_triage import EmailTriageEnv
from envs.data_cleaner import DataCleanerEnv
from envs.scheduler import SchedulerEnv
from core.model import EmailAction, CleanerAction, SchedulerAction

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

email_env = EmailTriageEnv(email_dataset)
cleaning_env = DataCleanerEnv(cleaning_dataset)
scheduling_env = SchedulerEnv(scheduling_dataset)

app = FastAPI()

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/reset")
def reset_all():
    email_env.reset()
    cleaning_env.reset()
    scheduling_env.reset()
    return {"status": "ok"}

@app.post("/step")
def step_all(action: dict):
    if "category" in action:
        return email_env.step(EmailAction(**action))
    elif "field" in action:
        return cleaning_env.step(CleanerAction(**action))
    elif "time" in action or "attendees" in action or "location" in action:
        return scheduling_env.step(SchedulerAction(**action))
    return {"error": "Unknown action"}

@app.get("/state")
def state_all():
    return {
        "email": email_env.current_email,
        "cleaning": cleaning_env.current_row,
        "scheduling": scheduling_env.current_row,
    }

@app.post("/email/reset")
def reset_email():
    return email_env.reset()

@app.post("/email/step")
def step_email(action: EmailAction):
    return email_env.step(action)

@app.post("/cleaning/reset")
def reset_cleaning():
    return cleaning_env.reset()

@app.post("/cleaning/step")
def step_cleaning(action: CleanerAction):
    return cleaning_env.step(action)

@app.post("/scheduling/reset")
def reset_scheduling():
    return scheduling_env.reset()

@app.post("/scheduling/step")
def step_scheduling(action: SchedulerAction):
    return scheduling_env.step(action)
