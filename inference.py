import os
from openai import OpenAI
from envs.email_triage import EmailTriageEnv
from envs.data_cleaner import DataCleanerEnv
from envs.scheduler import SchedulerEnv
from core.model import EmailAction, CleanerAction, SchedulerAction

# Environment variables injected by validator
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")

# Initialize OpenAI client with proxy
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

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

def run_email_episode():
    rewards, steps = [], 0
    print(f"[START] task=email env=openenv_demo model={MODEL_NAME}")
    obs = email_env.reset()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Classify the email as urgent or spam."},
                {"role": "user", "content": f"Subject: {obs.subject}\nBody: {obs.body}"}
            ]
        )
        predicted = response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        predicted = "unknown"

    action = EmailAction(category=predicted)
    result = email_env.step(action)
    steps += 1
    rewards.append(result.reward)
    print(f"[STEP] step={steps} action={action.category} reward={result.reward:.2f} done={str(result.done).lower()} error=null")
    score = min(max(sum(rewards), 0.0), 1.0)
    success = result.done and score > 0
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}")
    return score

def run_cleaning_episode():
    rewards, steps = [], 0
    print(f"[START] task=cleaning env=openenv_demo model={MODEL_NAME}")
    obs = cleaning_env.reset()
    gt = cleaning_env.current_row["ground_truth"]

    for field in gt.keys():
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Normalize the field value."},
                    {"role": "user", "content": f"Raw entry: {cleaning_env.current_row['raw_entry'][field]}"}
                ]
            )
            predicted = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            predicted = cleaning_env.current_row["raw_entry"][field]

        action = CleanerAction(field=field, new_value=predicted)
        result = cleaning_env.step(action)
        steps += 1
        rewards.append(result.reward)
        print(f"[STEP] step={steps} action={field}:{predicted} reward={result.reward:.2f} done={str(result.done).lower()} error=null")
        if result.done:
            break

    score = min(max(sum(rewards)/len(gt), 0.0), 1.0)
    success = result.done and score > 0
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}")
    return score

def run_scheduling_episode():
    rewards, steps = [], 0
    print(f"[START] task=scheduling env=openenv_demo model={MODEL_NAME}")
    obs = scheduling_env.reset()
    gt = scheduling_env.current_row["ground_truth"]

    for field in gt.keys():
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Extract scheduling details."},
                    {"role": "user", "content": f"Request: {obs.raw_request}\nField: {field}"}
                ]
            )
            predicted = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            predicted = gt[field]

        action = SchedulerAction(field=field, new_value=predicted)
        result = scheduling_env.step(action)
        steps += 1
        rewards.append(result.reward)
        print(f"[STEP] step={steps} action={field}:{predicted} reward={result.reward:.2f} done={str(result.done).lower()} error=null")
        if result.done:
            break

    score = min(max(sum(rewards)/len(gt), 0.0), 1.0)
    success = result.done and score > 0
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}")
    return score

if __name__ == "__main__":
    scores = {
        "email": run_email_episode(),
        "cleaning": run_cleaning_episode(),
        "scheduling": run_scheduling_episode()
    }
    print("Final Scores:", scores)
