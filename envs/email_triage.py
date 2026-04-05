from core.model import EmailAction, EmailObservation, EmailState
from core.graders import grade_email
import random 
import uuid

class EmailTriageEnv:
    def __init__(self,dataset):
        self.dataset = dataset
        self.current_email =None
        self.step_count = 0
        self.done =False
        self.episode_id= None

    def reset(self)->EmailObservation:
        self.current_email=random.choice(self.dataset)
        self.episode_id=str(uuid.uuid4())
        self.step_count=0
        self.done=False
        return EmailObservation(
            subject=self.current_email["subject"],
            body=self.current_email["body"],
            sender=self.current_email["sender"],
            timestamp=self.current_email["timestamp"]
        )

    def step(self, action: EmailAction)->dict:
        self.step_count+=1
        ground_truth= self.current_email["label"]
        reward = grade_email(action.category, ground_truth)
        
        self.done= True
        return {
            "observation": EmailObservation(
                subject=self.current_email["subject"],
                body=self.current_email["body"],
                sender=self.current_email["sender"],
                timestamp=self.current_email["timestamp"]
            ),
            "reward":reward,
            "done": self.done,
            "info": {"ground_truth":ground_truth}

        }   
    def state(self)-> EmailState:
        return EmailState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            done=self.done,
        )        

