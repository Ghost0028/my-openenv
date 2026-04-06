import random
import uuid
from core.model import CleanerAction, CleanerObservation, CleanerState
class DataCleanerEnv:
    def __init__(self,dataset):
        self.dataset=dataset
        self.current_row=None
        self.done=False
        self.step_count=0
        self.episode_id=None
        self.current_values={}

    def reset(self)-> CleanerObservation:
        self.current_row=random.choice(self.dataset)
        self.episode_id=str(uuid.uuid4())
        self.step_count=0
        self.done=False
        self.current_values=dict(self.current_row["raw_entry"])
        return CleanerObservation(
            current_values= self.current_values,
            reward= 0.0,
            done= self.done,
            info= {"ground_truth": self.current_row["ground_truth"]}
        )
       
    def step(self, action: CleanerAction)->dict:
        """
        action is expected to be a dict:
        {
            "field": <string>,   # which field to fix
            "new_value": <string> # agent’s proposed correction
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
        return CleanerObservation(
            current_values= self.current_values,
            reward= reward,
            done= self.done,
            info= {"ground_truth": ground_truth}
        )

    def state(self):
        return CleanerState(
            episode_id= self.episode_id,
            step_count= self.step_count,
            done= self.done
        )
