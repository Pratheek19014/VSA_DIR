import pandas as pd
import pandas.api.types
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Compute the mean absolute error (L1) for the few-shot region.
    The many-shot region must have a mean absolute error below 0.2, otherwise an error is raised.
    
    Parameters
    ----------
    solution : pd.DataFrame
        Ground truth dataframe with columns:
        - row_id_column_name: unique identifier for each row
        - ground_truth_sideslip: actual sideslip values
        - band: classification of shots ('many', 'medium', 'few')
    submission : pd.DataFrame
        Predictions dataframe with columns:
        - row_id_column_name: unique identifier for each row
        - predicted_sideslip: predicted sideslip values
    row_id_column_name : str
        Name of the column containing row identifiers (e.g., 'id')
    
    Returns
    -------
    float
        Mean absolute error for the few-shot region
    
    Raises
    ------
    ParticipantVisibleError
        - If submission has incorrect number of rows
        - If submission contains non-numeric values
        - If submission contains NaN values
        - If submission columns are incorrectly named
        - If many-shot MAE exceeds 0.2 threshold
    
    Examples
    --------
    >>> solution = pd.DataFrame({
    ...     'id': [0, 1, 2, 3, 4, 5],
    ...     'ground_truth_sideslip': [0.1, 0.2, 5.0, 5.5, 0.3, 6.0],
    ...     'band': ['many', 'many', 'few', 'few', 'many', 'few']
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1, 2, 3, 4, 5],
    ...     'predicted_sideslip': [0.11, 0.21, 5.5, 6.0, 0.31, 6.5]
    ... })
    >>> score(solution, submission, 'id')
    0.5
    Notes
    -----
    - The metric focuses on few-shot performance (rare sideslip values)
    - Many-shot predictions must maintain MAE < 0.2 as a quality gate
    - Medium-shot region is computed but not used in the final score
    - L1 (Mean Absolute Error) is used as the loss function
    '''

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if submission.shape[0] != solution.shape[0]:
        raise ParticipantVisibleError(f'Submission must have exactly {solution.shape[0]} rows, but it has {submission.shape[0]} rows instead.')

    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
        
    # check for NaNs
    if submission.isnull().values.any():
        raise ParticipantVisibleError('Submission contains NaN values, please replace them with a number')

    if list(submission.columns) != ["predicted_sideslip"]:
        raise ParticipantVisibleError(f'Submission columns must be named "predicted_sideslip", but they are named {list(submission.columns)} instead.')

    df = submission.merge(solution, left_index=True, right_index=True)
    df["L1"] = np.abs(df["predicted_sideslip"] - df["ground_truth_sideslip"])
    shot_errors = df.groupby("band")["L1"].mean()
    
    if float(shot_errors["many"]) > 0.2:
        raise ParticipantVisibleError(f'Your mean absolute error for the many shot region is {float(shot_errors["many"]):.3f}, which is above the threshold of 0.2. Please improve your model for these shots.')

    return shot_errors