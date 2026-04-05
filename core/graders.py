
def grade_email(action_category: str, ground_truth: str) -> float:
    """
    Returns a score between 0.0 and 1.0 based on how close the agent's
    classification is to the ground truth.
    """
    if action_category == ground_truth:
        return 1.0
    elif ground_truth == "urgent" and action_category == "informational":
        return 0.5  # partial credit
    elif ground_truth == "spam" and action_category == "informational":
        return 0.3  # weaker partial credit
    else:
        return 0.0
