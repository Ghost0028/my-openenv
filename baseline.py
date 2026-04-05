import random
from envs.email_triage import EmailTriageEnv
from envs.data_cleaner import DataCleanerEnv
from envs.scheduler import SchedulerEnv
from core.model import EmailAction

# Example datasets
email_dataset = [
    {
        "subject": "Meeting tomorrow",
        "body": "Don't forget the 10 AM meeting.",
        "sender": "boss@example.com",
        "timestamp": "2026-04-03T09:00:00",
        "label": "urgent"
    },
    {
        "subject": "Win a free iPhone!",
        "body": "Click here to claim your prize.",
        "sender": "scam@example.com",
        "timestamp": "2026-04-02T14:30:00",
        "label": "spam"
    }
]

cleaning_dataset = [
    {
        "raw_entry": {"name": "john DOE", "phone": "123-456-7890"},
        "clean_entry": {"name": "John Doe", "phone": "1234567890"}
    }
]

scheduling_dataset = [
    {
        "raw_request": "Meet John tomorrow afternoon in conference room",
        "ground_truth": {
            "date": "2026-04-06",
            "time": "15:00",
            "attendees": ["John"],
            "location": "conference room"
        }
    }
]

# Initialize environments
email_env = EmailTriageEnv(email_dataset)
cleaning_env = DataCleanerEnv(cleaning_dataset)
scheduling_env = SchedulerEnv(scheduling_dataset)

def run_email_baseline():
    obs = email_env.reset()
    ground_truth = email_env.current_email["label"]
    # Perfect agent: always picks the correct category
    action = EmailAction(category=ground_truth)
    result = email_env.step(action)
    return result["reward"]

def run_cleaning_baseline():
    obs = cleaning_env.reset()
    ground_truth = cleaning_env.current_row["clean_entry"]

    # Fill fields one by one with correct values
    for field, true_value in ground_truth.items():
        action = {"field": field, "new_value": true_value}
        result = cleaning_env.step(action)
        print(f"[Cleaner] Step {cleaning_env.step_count}: reward={result['reward']}, done={result['done']}")
        if result["done"]:
            break
    return result["reward"]

def run_scheduling_baseline():
    obs = scheduling_env.reset()
    ground_truth = scheduling_env.current_row["ground_truth"]

    # Fill fields one by one with correct values
    for field, true_value in ground_truth.items():
        action = {"field": field, "new_value": true_value}
        result = scheduling_env.step(action)
        print(f"[Scheduler] Step {scheduling_env.step_count}: reward={result['reward']}, done={result['done']}")
        if result["done"]:
            break
    return result["reward"]

if __name__ == "__main__":
    # Run multiple episodes for reproducibility
    episodes = 5
    scores = {"email_triage": [], "data_cleaning": [], "scheduling": []}

    for _ in range(episodes):
        scores["email_triage"].append(run_email_baseline())
        scores["data_cleaning"].append(run_cleaning_baseline())
        scores["scheduling"].append(run_scheduling_baseline())

    avg_scores = {task: sum(vals)/len(vals) for task, vals in scores.items()}
    print("\nBaseline average scores over", episodes, "episodes:")
    for task, score in avg_scores.items():
        print(f"{task}: {score:.2f}")
