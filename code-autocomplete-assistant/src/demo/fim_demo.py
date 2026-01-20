from typing import List, Dict


def process_user_data(users: List[Dict]) -> List[Dict]:
    """Process and validate user data.

    Args:
        users: List of user dictionaries

    Returns:
        List of validated user dictionaries
    """
    validated_users = []

    for user in users:
    # <CURSOR HERE - add validation logic>
    # The model should understand context from both
    # the prefix (function signature, docstring, loop)
    # and suffix (return statement)

    return validated_users


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a dataset."""
    if not data:
        return {}

    # <CURSOR HERE - calculate mean, median, std dev>

    return {
        "mean": mean,
        "median": median,
        "std_dev": std_dev
    }