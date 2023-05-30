from nflutils.dataprep import *

import pandas as pd
import numpy as np

from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":

    BASE_DIR = Path("data")
    
    labels = pd.read_csv(f"{BASE_DIR}/raw/train_labels.csv", parse_dates=["datetime"])

    tr_tracking = pd.read_csv(
        BASE_DIR/"raw/train_player_tracking.csv", parse_dates=["datetime"]
    )

    te_tracking = pd.read_csv(
        BASE_DIR/"raw/test_player_tracking.csv", parse_dates=["datetime"]
    )

    tr_helmets = pd.read_csv(BASE_DIR/"raw/train_baseline_helmets.csv")
    te_helmets = pd.read_csv(BASE_DIR/"raw/test_baseline_helmets.csv")

    tr_video_metadata = pd.read_csv(
        BASE_DIR/"raw/train_video_metadata.csv",
        parse_dates=["start_time", "end_time", "snap_time"],
    )

    df_combo = create_features(labels, tr_tracking)

    df_combo = df_combo[(df_combo.distance.isna()) | (df_combo.distance <= 2)]


    df_tracking_helmets = pd.DataFrame()
    for gp in tqdm(df_combo.game_play.unique()):
        df_tracking_helmets = pd.concat([df_tracking_helmets,
                                        merge_tracking_and_helmets_ts(df_combo, tr_helmets, tr_video_metadata, gp, 'Sideline'),
                                        merge_tracking_and_helmets_ts(df_combo, tr_helmets, tr_video_metadata, gp, 'Endzone')])
        
    df_tracking_helmets = calc_two_players_helmets_center(df_tracking_helmets)

    df_tracking_helmets.to_parquet(BASE_DIR+'/processed/df_tracking_helmets_below_2.parquet', index=False)