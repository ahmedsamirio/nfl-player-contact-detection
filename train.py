import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import pickle
import timm
import cv2

from tqdm.notebook import tqdm

from sklearn.metrics import matthews_corrcoef

import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastai.vision.all import *

import wandb
from fastai.callback.wandb import *

from nflutils.training import *
from nflutils.dataprep import *
from nflutils.validation import *

import torch

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--model', help='Model', type=str, default='convnext_tiny')
    parser.add_argument('--name', help='Experiment name', type=str)
    parser.add_argument('--fold', help='KFold', type=int, default=0)
    parser.add_argument('--seed', help='Seed', type=int, default=42)
    parser.add_argument('--type', help='Model Type', choices=['player-player', 'player-ground'], default='player-player')
    parser.add_argument('--log-wandb', help='Enable logging to WandB', action='store_true')
    parser.add_argument('--wandb-group', help='WandB group', type=str, default='')

    args = parser.parse_args()

    # Accessing the argument values
    model = args.model
    name = args.name
    fold = args.fold
    seed = args.seed
    type = args.type
    log_wandb = args.log_wandb
    wandb_group = args.wandb_group
    BASE_DIR = Path("data")
    frames_path = 'frames/content/work/frames/train'
    utils_path = 'nflutils'

    df_combo_with_helmets = pd.read_parquet(BASE_DIR+'/processed/df_tracking_helmets_below_thresh.parquet')
    df_combo_with_helmets['G_flag'] = np.where(df_combo_with_helmets.nfl_player_id_2 == 'G', 1, 0)
    kf_dict = pickle.load(open('kf_dict', 'rb'))

    if log_wandb:
        wandb.init(project='nfl-1st-and-future', group=wandb_group, name=name, force=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = A.Compose(
        [
            A.HorizontalFlip(),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=mean, std=mean),
            ToTensorV2(),
        ]
    )

    set_seed(seed, True)

    df = df_combo_with_helmets[(df_combo_with_helmets.left_2.notnull())].copy()

    frames_kwargs = dict(frames_path=frames_path, offset=0, sample_every_n_frame=6, undersample_no_contact=True, seed=seed)

    data = get_dls(df, kf_dict, fold, train_transform, val_transform, frames_kwargs=frames_kwargs, dl=NFLFrameTrackingDataset, bs=64)

    model = FrameTrackingModel(model)
    learn = Learner(data, model, CrossEntropyLossFlat(),
                    metrics=[accuracy, MatthewsCorrCoef(), Recall(), Precision(), F1Score()],
                    cbs=[
                            ShuffleGamePlayCallBack(df, kf_dict, 0, train_transform, val_transform, frames_kwargs, dl=NFLFrameTrackingDataset),
                            WandbCallback(log_preds=False, seed=seed),
                            SaveModelCallback(monitor='matthews_corrcoef', fname=name),
                            # GradientAccumulation(90)
                            # MixUp()
                    ]
                ).to_fp16()
    

    learn.fit_one_cycle(10, 1e-3)
    torch.save(learn.model, f'models/{name}.pkl')

    val_games = kf_dict[fold]['val_games']

    val_df = df_combo_with_helmets.query('game_play in @val_games').copy()
    val_df = val_df[(val_df.left_2.notnull())].copy()
    val_df['frame'] = val_df['frame'].astype(int)

    test_ds = NFLFrameTrackingDataset(val_df, transform=val_transform, crop_size=256)

    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True,
    )

    preds, _ = learn.get_preds(dl=test_loader)

    val_df['contact_pred'] = preds[:, 1]

    val_df.to_parquet(BASE_DIR+f'/validation/{name}.parquet', index=False)