import pandas as pd


def get_session_metrics(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Given a pandas DataFrame in the format of the train dataset and a user_id, return the following metrics for every session_id of the user:
        - user_id (int) : the given user id.
        - session_id (int) : the session id.
        - total_session_time (float) : The time passed between the first and last interactions, in seconds. Rounded to the 2nd decimal.
        - cart_addition_ratio (float) : Percentage of the added products out of the total products interacted with. Rounded ot the 2nd decimal.

    If there's no data for the given user, return an empty Dataframe preserving the expected columns.
    The column order and types must be scrictly followed.

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame  of the data to be used for the agent.
    user_id : int
        Id of the client.

    Returns
    -------
    Pandas Dataframe with some metrics for all the sessions of the given user.
    """
    user_df = df[df["user_id"] == user_id].copy()
    
    if user_df.empty:
        return pd.DataFrame(columns=["user_id", "session_id", "total_session_time", "cart_addition_ratio"])
    
    user_df["timestamp_local"] = pd.to_datetime(user_df["timestamp_local"], errors="coerce")
    
    user_df = user_df.dropna(subset=["timestamp_local"])
    
    session_duration = user_df.groupby("session_id")["timestamp_local"].agg(["min", "max"])
    session_duration["total_session_time"] = (session_duration["max"] - session_duration["min"]).dt.total_seconds().fillna(0)
    
    session_data = user_df.groupby("session_id").agg(
        total_interactions=("partnumber", "count"),
        cart_additions=("add_to_cart", "sum")
    )
    
    session_data["cart_addition_ratio"] = (
        session_data["cart_additions"] / session_data["total_interactions"]
    ).fillna(0).round(2)
    
    result = session_duration[["total_session_time"]].merge(
        session_data[["cart_addition_ratio"]],
        left_index=True,
        right_index=True
    )
    
    result = result.reset_index()
    result["user_id"] = user_id
    result["total_session_time"] = result["total_session_time"].round(2)
    
    result = result[["user_id", "session_id", "total_session_time", "cart_addition_ratio"]]
    
    result = result.sort_values(by=["user_id", "session_id"]).reset_index(drop=True)
    
    return result
    ...
