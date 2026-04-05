from dataclasses import dataclass
from typing import Literal

@dataclass
class EmailAction:
    category: Literal["urgent","personal","spam","informational"]

@dataclass
class EmailObservation:   
    subject: str
    body: str
    sender: str
    timestamp: str

@dataclass
class EmailState:
    episode_id: str
    step_count: int
    done: bool    

# Dataclasses are not needed for the datacleaner and scheduler since we are planning to use dict to give us more freedom

