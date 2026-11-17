QUESTION_ID = "Grading ID"
GROUND_TRUTH_KEY = "Points"
from utils.experiment import TASK_MODEL

MODEL = TASK_MODEL

def format_input_dict(row):
    # Extract the inputs for the task from the row
    return {
        "domain": "imo_grading",
        "problem": row['Problem'],
        "solution": row['Solution'],
        "grading_guidelines": row['Grading guidelines'],
        "student_answer": row['Response'],
    }

VALID_POINTS = (0, 1, 6, 7)


def normalize_points(value):
    """Return a valid integer score, or None for missing/invalid predictions."""
    from decimal import Decimal, InvalidOperation

    if isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).strip())
        if number.is_finite() and number in VALID_POINTS:
            return int(number)
    except (InvalidOperation, ValueError, TypeError):
        pass
    return None
