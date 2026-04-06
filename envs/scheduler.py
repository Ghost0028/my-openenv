import random 
import uuid
from core.model import SchedulerAction, SchedulerObservation, SchedulerState
class SchedulerEnv:
    def __init__(self,dataset):
        self.dataset= dataset
        self.current_row=None
        self.step_count =0
        self.done=False
        self.episode_id =None
        self.current_values={}
        
    def reset(self)->SchedulerObservation:
        self.current_row=random.choice(self.dataset)
        self.episode_id=str(uuid.uuid4())
        self.step_count=0
        self.done=False  
        self.raw_string=self.current_row["raw_request"]
        self.current_values={}
        return SchedulerObservation(
            raw_string=self.raw_string,
            current_values=self.current_values,
            done=self.done,
            info= {"ground_truth": self.current_row["ground_truth"]},
            reward=0.0
        )  
    
    def step(self, action: SchedulerAction)->SchedulerObservation:
        """
        action is expected to be a dict:
        {
            "field": <string>,   # which field to fix
            "new_value": <string> # agent’s proposed answer
        }
        """
        self.step_count+=1
        ground_truth = self.current_row["ground_truth"]
        field = action.field
        self.current_values[field]=action.new_value
        correct_fields=sum(
            1 for k in ground_truth
            if self.current_values.get(k)==ground_truth[k]
        )  
        reward = correct_fields/len(ground_truth)

        if reward ==1.0:
            self.done=True
        return SchedulerObservation(
            raw_string=self.raw_string,
            current_values=self.current_values,
            done=self.done,
            info= {"ground_truth": ground_truth},
            reward=reward
        )  

    def state(self):
        return  SchedulerState(
            episode_id= self.episode_id,
            step_count= self.step_count,
            done= self.done
        ) 
