import pandas as pd
import numpy as np

import cv2

from fastai.vision.all import *

from sklearn.metrics import matthews_corrcoef



## Source: https://www.kaggle.com/code/robikscube/nfl-player-contact-detection-getting-started
def compute_distance(df, tr_tracking, merge_col="datetime"):
    """
    Merges tracking data on player1 and 2 and computes the distance.
    """
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id", "x_position", "y_position"]
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={"x_position": "x_position_1", "y_position": "y_position_1"})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id", "x_position", "y_position"]
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={"x_position": "x_position_2", "y_position": "y_position_2"})
        .copy()
    )

    df_combo["distance"] = np.sqrt(
        np.square(df_combo["x_position_1"] - df_combo["x_position_2"])
        + np.square(df_combo["y_position_1"] - df_combo["y_position_2"])
    )
    return df_combo

## Source: https://www.kaggle.com/code/robikscube/nfl-player-contact-detection-getting-started
def add_contact_id(df):
    # Create contact ids
    df["contact_id"] = (
        df["game_play"]
        + "_"
        + df["step"].astype("str")
        + "_"
        + df["nfl_player_id_1"].astype("str")
        + "_"
        + df["nfl_player_id_2"].astype("str")
    )
    return df

## Source: https://www.kaggle.com/code/robikscube/nfl-player-contact-detection-getting-started
def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df

def merge_tracking_and_helmets(tracking_df, helmets_df):
    # Create a dict that maps frames to steps in order to merge tracking data with helmets data
    frame_step_map = {}
    for s in range(0, 200):
        for f in range(289 + s*6, 289 + s*6 + 7):
            frame_step_map[f] = s
    
    # Add the step column to helmets data
    helmets_df = (helmets_df
                    .assign(step=helmets_df.frame.map(frame_step_map))
                    .astype({'nfl_player_id': 'str', 'step': 'int'}))
    
    # Merge the data
    return (tracking_df
        .astype({"nfl_player_id_1": "str"})
        .merge(helmets_df.astype({"nfl_player_id": "str"})[
                ['game_play', 'view', 'step', 'frame', 'nfl_player_id', 'player_label', 'left', 'width', 'top', 'height']
               ], 
               left_on=['game_play', 'step', 'nfl_player_id_1'], 
               right_on=['game_play', 'step', 'nfl_player_id'],
               how='left')
         .rename(columns={"player_label": "player_label_1",
                          "left": "left_1",
                          "width": "width_1",
                          "top": "top_1",
                          "height": "height_1"})
         .drop(["nfl_player_id"], axis=1)
         .astype({"nfl_player_id_2": "str"})
         .merge(helmets_df.astype({"nfl_player_id": "str"})[
                 ['game_play', 'view', 'step', 'frame', 'nfl_player_id', 'player_label', 'left', 'width', 'top', 'height']
               ], 
               left_on=['game_play', 'step', 'frame', 'view', 'nfl_player_id_2'], 
               right_on=['game_play', 'step', 'frame', 'view', 'nfl_player_id'],
               how='left')
         .rename(columns={"player_label": "player_label_2",
                          "left": "left_2",
                          "width": "width_2",
                          "top": "top_2",
                          "height": "height_2"})
         .drop(["nfl_player_id"], axis=1)
    )

def merge_tracking_and_helmets_ts(tracking_df, helmets_df, meta_df, game_play, view, fps=59.94):
    gp_track = tracking_df.query('game_play == @game_play').copy()
    gp_helms = helmets_df.query('game_play == @game_play and view == @view').copy()
    
    start_time = meta_df.query("game_play == @game_play and view == @view")[
        "start_time"
    ].values[0]
    
    gp_helms["datetime"] = (
        pd.to_timedelta(gp_helms["frame"] * (1 / fps), unit="s") + start_time
    )
    gp_helms["datetime"] = pd.to_datetime(gp_helms["datetime"], utc=True)
    gp_helms["datetime_ngs"] = (
        pd.DatetimeIndex(gp_helms["datetime"] + pd.to_timedelta(50, "ms"))
        .floor("100ms")
        .values
    )
    gp_helms["datetime_ngs"] = pd.to_datetime(gp_helms["datetime_ngs"], utc=True)

    gp_track["datetime_ngs"] = pd.to_datetime(gp_track["datetime"], utc=True)
    
    # Merge the data
    return (gp_track
        .astype({"nfl_player_id_1": "str"})
        .merge(gp_helms.astype({"nfl_player_id": "str"})[
                ['game_play', 'view', 'datetime_ngs', 'frame', 'nfl_player_id', 'player_label', 'left', 'width', 'top', 'height']
               ], 
               left_on=['game_play', 'datetime_ngs', 'nfl_player_id_1'], 
               right_on=['game_play', 'datetime_ngs', 'nfl_player_id'],
               how='left')
         .rename(columns={"player_label": "player_label_1",
                          "left": "left_1",
                          "width": "width_1",
                          "top": "top_1",
                          "height": "height_1"})
         .drop(["nfl_player_id"], axis=1)
         .astype({"nfl_player_id_2": "str"})
         .merge(gp_helms.astype({"nfl_player_id": "str"})[
                 ['game_play', 'view', 'datetime_ngs', 'frame', 'nfl_player_id', 'player_label', 'left', 'width', 'top', 'height']
               ], 
               left_on=['game_play', 'datetime_ngs', 'frame', 'view', 'nfl_player_id_2'], 
               right_on=['game_play', 'datetime_ngs', 'frame', 'view', 'nfl_player_id'],
               how='left')
         .rename(columns={"player_label": "player_label_2",
                          "left": "left_2",
                          "width": "width_2",
                          "top": "top_2",
                          "height": "height_2"})
         .drop(["nfl_player_id"], axis=1)
    )

def calc_two_players_helmets_center(df_combo):
    df_combo = df_combo.assign(
        top=df_combo[['top_1', 'top_2']].min(axis=1),
        left=df_combo[['left_1', 'left_2']].min(axis=1),
        bottom=pd.concat([
                df_combo[['top_1', 'height_1']].sum(axis=1),
                df_combo[['top_2', 'height_2']].sum(axis=1),
            ], axis=1).max(axis=1),
        right=pd.concat([
                df_combo[['left_1', 'width_1']].sum(axis=1),
                df_combo[['left_2', 'width_2']].sum(axis=1),
            ], axis=1).max(axis=1),
    )
    
    df_combo = df_combo.assign(
        center_y = (df_combo.top + df_combo.bottom) // 2,
        center_x = (df_combo.left + df_combo.right) // 2
    )
    
    df_combo['center_x'] = np.where(df_combo.center_x.isna(),
                                    df_combo[['left_1', 'width_1']].sum(axis=1) // 2,
                                    df_combo.center_x)
    
    df_combo['center_y'] = np.where(df_combo.center_x.isna(),
                                    df_combo[['top_1', 'height_1']].sum(axis=1) // 2,
                                    df_combo.center_y)
    
    return df_combo



def validate_model(learn, df_combo, val_df, thresh=0.3):
    val_dl = learn.dls.test_dl(val_df, with_labels=True)
    preds, _ = learn.get_preds(dl=val_dl) 
    
    val_df.loc[:, 'contact_pred'] = preds.cpu().detach().numpy()[:, 1]
    val_df = add_contact_id(val_df)
    
    val_dist = df_combo[df_combo.game_play.isin(val_df.game_play.unique())].copy()
    
    val_dist["distance"] = val_dist["distance"].fillna(99)  # Fill player to ground with 9    
    val_dist_agg = val_dist.merge(val_df.groupby('contact_id', as_index=False).contact_pred.mean(), how='left', on='contact_id')
    
    out = np.where(val_dist_agg['contact_pred'].isna(),
                   val_dist['distance'] <= 1, 
                   val_dist_agg['contact_pred'] > thresh).astype(int)
    
    print('Baseline', matthews_corrcoef(val_dist_agg['contact'], (val_dist_agg['distance'] <= 1).astype(int)))
    print('Model', matthews_corrcoef(val_dist_agg['contact'], out))
    
    return val_dist