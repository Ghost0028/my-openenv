from dataclasses import dataclass
from typing import Literal, Any, Dict

# Email
@dataclass
class EmailAction:
    category: Literal["urgent","personal","spam","informational"]

@dataclass
class EmailObservation:   
    subject: str
    body: str
    sender: str
    timestamp: str
    reward: float
    done: bool
    info: Dict[str, Any]

@dataclass
class EmailState:
    episode_id: str
    step_count: int
    done: bool    

# Cleaner
@dataclass
class CleanerAction:
    field: str
    new_value: str

@dataclass
class CleanerObservation:   
    current_values: Dict[str, str]
    reward: float
    done: bool
    info: Dict[str, str]

@dataclass
class CleanerState:
    episode_id: str
    step_count: int
    done: bool    

# Scheduler
@dataclass
class SchedulerAction:
    field: str
    new_value: Any

@dataclass
class SchedulerObservation:   
    raw_string: str
    current_values: Dict[str,Any]
    reward: float
    done: bool
    info: Dict[str, Any]

@dataclass
class SchedulerState:
    episode_id: str
    step_count: int
    done: bool    
