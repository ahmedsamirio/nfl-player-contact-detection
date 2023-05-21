import pandas as pd
import numpy as np

import cv2

from fastai.vision.all import *

from sklearn.metrics import matthews_corrcoef

from .dataprep import get_label

from functools import lru_cache

import os


@lru_cache(maxsize=1000)
def read_img(path):
    if path[-4:] == '.npy':
        return np.load(path)
    else:
        frame = cv2.imread(path)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def get_frame_path(row, frames_path, window, window_step, fmt):
    min_range = row['frame'] if window <= 6 else row['frame']-(window//3)
    max_range = row['frame'] + window if window <= 6 else row['frame']+(window-(window//3))
    frames = range(min_range, max_range, window_step)
    paths = []
    for frame in frames:
        frame = frame if len(str(frame)) > 3 else f'0{frame}'
        game_play = row['game_play']
        view = row['view']
        paths.append(f'{frames_path}/{game_play}_{view}.mp4_{frame}.{fmt}')
    return paths

def get_label(row):
    return ['no contact', 'contact'][row.contact]

def get_img(row, img_size=128, add_helmets=True, axis=0):
    frames = []
    for path in row.path:
        if not os.path.isfile(path):
            continue
            
        frame = read_img(path)

        if add_helmets:
            frame = cv2.rectangle(frame, 
                                  (int(row.left_1), int(row.top_1)),
                                  (int(row.left_1+row.width_1), int(row.top_1+row.height_1)),
                                  (255, 0, 0), 1)

            frame = cv2.rectangle(frame, 
                                  (int(row.left_2), int(row.top_2)),
                                  (int(row.left_2+row.width_2), int(row.top_2+row.height_2)),
                                  (255, 0, 0), 1)

        size = img_size // 2

        if row.center_y-size < 0:
            min_y = 0
            max_y = min_y + 2*size

        elif row.center_y+size > 719:
            min_y = 719 - 2*size
            max_y = 719

        else:
            min_y = row.center_y - size
            max_y = row.center_y + size

        if row.center_x-size < 0:
            min_x = 0
            max_x = min_x + 2*size

        elif row.center_x+size > 1279:
            min_x = 1279 - 2*size
            max_x = 1279

        else:
            min_x = row.center_x - size
            max_x = row.center_x + size
            
        if len(frame.shape) > 2:
            frame = frame[int(min_y):int(max_y), 
                                  int(min_x):int(max_x), :]
        else:
            frame = frame[int(min_y):int(max_y), 
                                  int(min_x):int(max_x)]
        
        frames.append(frame)
    
    if len(frames) < len(row.path):
        if os.path.isfile(row.path[0]):
            frames = frames + [frames[-1]]*(len(row.path) - len(frames))
        else:
            frames = [frames[0]]*(len(row.path) - len(frames)) + frames
            
    if axis == 0:
        return (np.stack(frames, axis=0) / 255).astype(np.float32)
    else:
        return np.stack(frames, axis=-1)


def get_frames_df(df_combo, kf_dict, split, frames_path, window=3, window_step=2, sample_every_n_frame=None, sample_train=None, sample_val=None, undersample_no_contact=False, filter_views=None, train_fmt='jpg', valid_fmt='jpg', seed=42):
    train_game_plays = kf_dict[split]['train_games']
    val_game_plays = kf_dict[split]['val_games']
    
    train_combo = df_combo.query('game_play in @train_game_plays').copy()
    val_combo = df_combo.query('game_play in @val_game_plays').copy()
    
    train_combo['is_valid'] = False
    val_combo['is_valid'] = True
    
    if sample_every_n_frame is not None:
        train_combo = train_combo.query('(290 - frame) % @sample_every_n_frame == 0')
        val_combo = val_combo.query('(290 - frame) % @sample_every_n_frame == 0')
    
    if sample_train is not None:
        train_combo = train_combo.sample(frac=sample_train, random_state=seed)
        
    if sample_val is not None:
        val_combo = val_combo.sample(frac=sample_val, random_state=seed)
        
    if undersample_no_contact:
        train_combo = pd.concat([
            train_combo.query('contact == 1'),
            train_combo.query('contact == 0').sample(
                len(train_combo.query('contact == 1')), random_state=seed
            )
        ])
        
    train_combo.frame = train_combo.frame.astype('int')
    val_combo.frame = val_combo.frame.astype('int')
        
    train_combo['path'] = train_combo.apply(lambda x: get_frame_path(x, frames_path, window, window_step, fmt=train_fmt), axis=1)
    val_combo['path'] = val_combo.apply(lambda x: get_frame_path(x, frames_path, window, window_step, fmt=valid_fmt), axis=1)
    
    frames_df = pd.concat([train_combo, val_combo], axis=0)
    
    if filter_views is not None:
        frames_df = frames_df.query('view in @filter_views')
            
    return frames_df


def get_3d_dls(frames_df, img_size=128, item_tfms=None, batch_tfms=None, bs=64, shuffle=True, drop_last=False):
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=partial(get_img, img_size=img_size, add_helmets=True, axis=1),
        get_y=get_label,
        splitter=ColSplitter(),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    ).dataloaders(frames_df, bs=bs, shuffle=shuffle, drop_last=drop_last)


def get_nd_dls(frames_df, img_size=224, bs=32, sample=False, n=5):
    tfms = [[partial(get_img, img_size=img_size)],
    [get_label, Categorize(vocab=['contact', 'no contact'])]]

    means = [0.485, 0.456, 0.406] * 10
    stds = [0.229, 0.224, 0.225] * 10
    
    means = means[:n]
    stds = stds[:n]

    after_item = [ToTensor()]
    after_batch = [*aug_transforms(), Normalize.from_stats(means, stds)]

    splits = ColSplitter()(frames_df)
    dsets = Datasets(frames_df, tfms, splits=splits)

    return dsets.dataloaders(bs=bs, after_item=after_item, after_batch=after_batch, shuffle=True)


def get_learner(model, dls, n_in=3, metrics=[accuracy, F1Score]):
    return vision_learner(dls, model, metrics=metrics, n_in=n_in).to_fp16()