import pandas as pd
import numpy as np

import cv2

from fastai.vision.all import *

from sklearn.metrics import matthews_corrcoef

from torch.utils.data import Dataset, DataLoader
import functools

from pathlib import Path

def add_helmets(frame, row):
    frame = frame.copy()
    frame = cv2.rectangle(frame, 
                          (int(row.left_1), int(row.top_1)),
                          (int(row.left_1+row.width_1), int(row.top_1+row.height_1)),
                          (255, 0, 0), 2)
    
    if not np.isnan(row.left_2):
        frame = cv2.rectangle(frame, 
                              (int(row.left_2), int(row.top_2)),
                              (int(row.left_2+row.width_2), int(row.top_2+row.height_2)),
                              (255, 0, 0), 2)
    return frame

def add_helmet(frame, row):
    frame = frame.copy()
    frame = cv2.rectangle(frame, 
                          (int(row.left), int(row.top)),
                          (int(row.left+row.width), int(row.top+row.height)),
                          (255, 0, 0), 2)
    return frame


@functools.lru_cache(maxsize=500)
def _get_frame(path):
    frame = cv2.imread(path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def get_frame_path(frame_no, frames_path, game_play, view):
    frame = frame_no if len(str(frame_no)) > 3 else f'0{frame_no}'
    return f'{frames_path}/{game_play}_{view}.mp4_{frame}.jpg'


def get_frames_df(df_combo, kf_dict, split, frames_path, offset=0, sample_every_n_frame=None, sample_every_n_frame_train=None, sample_every_n_frame_val=None, sample_train=None, sample_val=None, undersample_no_contact=False, filter_views=None, seed=42):
    
    set_seed(seed, True)
    
    train_game_plays = kf_dict[split]['train_games']
    val_game_plays = kf_dict[split]['val_games']
    
    train_combo = df_combo.query('game_play in @train_game_plays').copy()
    val_combo = df_combo.query('game_play in @val_game_plays').copy()
    
    train_combo['is_valid'] = False
    val_combo['is_valid'] = True
    
    if sample_every_n_frame is not None:
        train_combo = train_combo.query('(290 - frame - @offset) % @sample_every_n_frame == 0')
        val_combo = val_combo.query('(290 - frame) % @sample_every_n_frame == 0')
        
    if sample_every_n_frame_train is not None:
        train_combo = train_combo.query('(290 - frame -@offset) % @sample_every_n_frame_train == 0')
        
    if sample_every_n_frame_val is not None:
        val_combo = val_combo.query('(290 - frame) % @sample_every_n_frame_val == 0')
    
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

    frames_df = pd.concat([train_combo, val_combo], axis=0)
    frames_df.frame = frames_df.frame.astype('int') 
    
    if filter_views is not None:
        frames_df = frames_df.query('view in @filter_views')
        
    # frames_df['path'] = frames_df.apply(lambda x: get_frame_path(x, frames_path), axis=1)
        
    return frames_df

def get_interpolated_player_helmets(helmets_df, player_helmets_df, tr_helmets, player_id):
    player_helmets = pd.DataFrame()
    tmp_frames_df = pd.DataFrame({'frame': np.arange(player_helmets_df.frame.min(), tr_helmets.frame.max()+1)})

    for view in helmets_df.view.unique():
        player_helmets_view = helmets_df.query('nfl_player_id == @player_id and view == @view')
        player_helmets_view = pd.merge(tmp_frames_df, player_helmets_view[['frame', 'left', 'width', 'top', 'height']], how='left', on='frame').interpolate(limit_direction='both')
        player_helmets_view['view'] = view
        player_helmets = pd.concat([player_helmets, player_helmets_view], axis=0)
        
    return player_helmets

def get_img(row, img_size=128, add_helmets=True):
    
    frame = cv2.imread(row.path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if add_helmets:
        frame = cv2.rectangle(frame, 
                              (int(row.left_1), int(row.top_1)),
                              (int(row.left_1+row.width_1), int(row.top_1+row.height_1)),
                              (255, 0, 0), 2)
        
        frame = cv2.rectangle(frame, 
                              (int(row.left_2), int(row.top_2)),
                              (int(row.left_2+row.width_2), int(row.top_2+row.height_2)),
                              (255, 0, 0), 2)
    
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
        
    cropped_frame = frame[int(min_y):int(max_y), 
                          int(min_x):int(max_x), :]
    return cropped_frame

def get_label(row):
    return ['no contact', 'contact'][row.contact]


def get_rgb_dls(frames_df, img_size=128, item_tfms=None, batch_tfms=None, bs=64, shuffle=True, drop_last=False):
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=partial(get_img, img_size=img_size),
        get_y=get_label,
        splitter=ColSplitter(),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    ).dataloaders(frames_df, bs=bs, shuffle=shuffle, drop_last=drop_last)


def get_learner(model, dls):
    return vision_learner(dls, model, metrics=accuracy)

class NFLFrameDataset(Dataset):
    def __init__(self, frames_df, transform=None, crop_size=256, helmets=True):
        self.frames_df = frames_df
        self.helmets = helmets
        self.crop_size = crop_size
        self.transform = transform
        
    def __len__(self):
        return len(self.frames_df)
        
    def __getitem__(self, idx):
        row = self.frames_df.iloc[idx]
        frame_path = get_frame_path(int(row.frame), CFG.frames_path, row['game_play'], row['view'])
        frame = self.get_frame(frame_path)
        if self.helmets:
            frame = add_helmets(frame, row)
            # frame = add_helmet_heatmap(frame, row)
            
        frame = crop_frame(frame, row.center_x, row.center_y, self.crop_size)
        if self.transform is not None:
            frame = self.transform(image=frame)['image']
        return frame, row.contact
    
    def get_frame(self, path):
        return _get_frame(path)


class NFLFrameTrackingDataset(Dataset):
    def __init__(self, frames_df, transform=None, crop_size=256, helmets=True):
        self.frames_df = frames_df
        self.helmets = helmets
        self.crop_size = crop_size
        self.transform = transform
        self.tracking_cols = ["x_position_1", "y_position_1", "speed_1", "distance_1", "direction_1", "orientation_1", "acceleration_1", "sa_1",
                              "x_position_2", "y_position_2", "speed_2", "distance_2", "direction_2", "orientation_2", "acceleration_2", "sa_2",
                              "distance", "G_flag"]
        
    def __len__(self):
        return len(self.frames_df)
        
    def __getitem__(self, idx):
        row = self.frames_df.iloc[idx]
        frame_path = get_frame_path(int(row.frame), CFG.frames_path, row['game_play'], row['view'])
        frame = self.get_frame(frame_path)
        
        if self.helmets:
            frame = add_helmets(frame, row)
            # frame = add_helmet_heatmap(frame, row)
            
        frame = crop_frame(frame, row.center_x, row.center_y, self.crop_size)
        
        if self.transform is not None:
            frame = self.transform(image=frame)['image']
            
        tracking_data = row[self.tracking_cols].fillna(-1).values.astype(np.float32)
        
        return frame, tracking_data, row.contact
    
    def get_frame(self, path):
        return _get_frame(path)
    

def get_interpolated_player_helmets(helmets_df, tmp_frames_df, player_id, view):
    player_helmets_view = helmets_df.query('nfl_player_id == @player_id and view == @view')
    player_helmets_view = pd.merge(tmp_frames_df, player_helmets_view[['frame', 'left', 'width', 'top', 'height']], how='left', on='frame').interpolate(limit_direction='both')
        
    return player_helmets_view

class NFL25DDataset(Dataset):
    def __init__(self, frames_df, transform=None, crop_size=256, helmets=True, frames_offsets=[-6, 0, 6]):
        self.frames_df = frames_df
        self.helmets = helmets
        self.crop_size = crop_size
        self.transform = transform
        self.tracking_cols = ["x_position_1", "y_position_1", "speed_1", "distance_1", "direction_1", "orientation_1", "acceleration_1", "sa_1",
                              "x_position_2", "y_position_2", "speed_2", "distance_2", "direction_2", "orientation_2", "acceleration_2", "sa_2",
                              "distance", "G_flag"]
        
        self.gps_helmet_dfs = {gp: tr_helmets.query('game_play == @gp') for gp in frames_df.game_play.unique()}
        self.frames_offsets = frames_offsets
        self.tmp_frames_df = pd.DataFrame({'frame': np.arange(270, 1500)})


        
    def __len__(self):
        return len(self.frames_df)
        
    def __getitem__(self, idx):
        row = self.frames_df.iloc[idx]
        gp = row['game_play']
        view = row['view']
        
        player_1_id = int(row.nfl_player_id_1)
        player_2_id = int(row.nfl_player_id_2)
        
        player_1_helmets = get_interpolated_player_helmets(self.gps_helmet_dfs[gp], self.tmp_frames_df, player_1_id, view)
        player_2_helmets = get_interpolated_player_helmets(self.gps_helmet_dfs[gp], self.tmp_frames_df, player_2_id, view)
        
        frames = []
                
        for frame_offset in [-6, 0, 6]:
            frame_no = int(row.frame + frame_offset)
            frame_path = get_frame_path(frame_no, CFG.frames_path, gp, view)
            
            if os.path.exists(frame_path):
                frame = self.get_frame(frame_path)

                if self.helmets:
                    try:
                        player_1_frame_helmet = player_1_helmets.query('frame == @frame_no').iloc[0]
                        player_2_frame_helmet = player_2_helmets.query('frame == @frame_no').iloc[0]

                        frame = add_helmet(frame, player_1_frame_helmet)
                        frame = add_helmet(frame, player_2_frame_helmet)
                    except:
                        print(gp, player_1_id, view, frame_no)

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                frames.append(frame)
            
        if len(frames) < len(self.frames_offsets):
            frames = frames + frames[-1:]*(len(self.frames_offsets)-len(frames))
            
        frame = np.stack(frames, axis=-1)
        frame = crop_frame(frame, row.center_x, row.center_y, self.crop_size)
        
        if self.transform is not None:
            frame = self.transform(image=frame)['image']
        
        tracking_data = row[self.tracking_cols].fillna(-1).values.astype(np.float32)
        
        return frame, tracking_data, row.contact
            
    def get_frame(self, path):
        return _get_frame(path)
    
def crop_frame(frame, x, y, size):
    size = size // 2
    
    if y-size < 0:
        min_y = 0
        max_y = min_y + 2*size
        
    elif y+size > 719:
        min_y = 719 - 2*size
        max_y = 719
        
    else:
        min_y = y - size
        max_y = y + size
    
    if x-size < 0:
        min_x = 0
        max_x = min_x + 2*size
        
    elif x+size > 1279:
        min_x = 1279 - 2*size
        max_x = 1279
        
    else:
        min_x = x - size
        max_x = x + size
        
    cropped_frame = frame[int(min_y):int(max_y), 
                          int(min_x):int(max_x), :]
    return cropped_frame

def crop_frames(frames, x, y, size):
    size = size // 2
    
    if y-size < 0:
        min_y = 0
        max_y = min_y + 2*size
        
    elif y+size > 719:
        min_y = 719 - 2*size
        max_y = 719
        
    else:
        min_y = y - size
        max_y = y + size
    
    if x-size < 0:
        min_x = 0
        max_x = min_x + 2*size
        
    elif x+size > 1279:
        min_x = 1279 - 2*size
        max_x = 1279
        
    else:
        min_x = x - size
        max_x = x + size
        
    cropped_frames = frames[:,
                            int(min_y):int(max_y), 
                            int(min_x):int(max_x),
                            :]
    return cropped_frames

def get_dls(df_combo, kf_dict, split, train_transform, val_transform, frames_kwargs, crop_size=256, bs=64, num_workers=8, dl=NFLFrameTrackingDataset):
    frames_df = get_frames_df(df_combo, kf_dict, split, **frames_kwargs).copy()
    
    train_frames_df = frames_df.query('~is_valid').copy().reset_index(drop=True)
    val_frames_df = frames_df.query('is_valid').copy().reset_index(drop=True)
    
    # train_frames_df.loc[(train_frames_df.nfl_player_id_2 == "G") & (train_frames_df.contact == 1)] = 2
    # val_frames_df.loc[(val_frames_df.nfl_player_id_2 == "G") & (val_frames_df.contact == 1)] = 2
    
    set_seed(frames_kwargs['seed'], True)

    train_ds = dl(train_frames_df, transform=train_transform, helmets=True, crop_size=crop_size)
    val_ds = dl(val_frames_df, transform=val_transform, helmets=True, crop_size=crop_size)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    
    data = DataLoaders(train_loader, val_loader, device=torch.device('cuda'))
    
    return data


def visualize_batch(batch, means, stds):
    """
    Visualize a batch of image data using Matplotlib.
    
    Parameters:
        - batch_tensor (torch.Tensor): A batch of image data in the shape (batch_size, channels, height, width).
        - means (tuple): A tuple of means for each channel in the image data.
        - stds (tuple): A tuple of standard deviations for each channel in the image data.
    """
    # Make a copy of the batch tensor so that we don't modify the original data
    batch_tensor = batch[0].clone()
    
    # De-normalize the data using the means and stds
    batch_tensor = batch_tensor * torch.tensor(stds, dtype=batch_tensor.dtype).view(3, 1, 1) + torch.tensor(means, dtype=batch_tensor.dtype).view(3, 1, 1)
    
    # Convert the data to numpy and transpose it to (batch_size, height, width, channels)
    batch_np = batch_tensor.numpy().transpose(0, 2, 3, 1)
    
    # Plot the images
    plt.figure(figsize=(20, 20))
    for i in range(batch_np.shape[0]):
        plt.subplot(batch_np.shape[0] // 5 + 1, 5, i + 1)
        plt.imshow(batch_np[i])
        plt.title(['no contact', 'contact'][batch[-1][i].item()])
        plt.axis("off")
    plt.show()



class ShuffleGamePlayCallBack(Callback):
    def __init__(self, df_combo, kf_dict, split, train_transform, val_transform, frames_kwargs, crop_size=256, bs=64, num_workers=8, dl=NFLFrameTrackingDataset):
        self.df_combo = df_combo
        self.kf_dict = kf_dict
        self.split = split
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.crop_size = crop_size
        self.bs = bs
        self.num_workers = 8
        self.frames_kwargs = frames_kwargs
        self.dl = dl
        
    def after_epoch(self):
        self.frames_kwargs['seed'] += 1
        self.frames_kwargs['offset'] += 1
        print(self.frames_kwargs['seed'], self.frames_kwargs['offset'])
        self.learn.dls = get_dls(self.df_combo, self.kf_dict, self.split, self.train_transform,
                                 self.val_transform, self.frames_kwargs, self.crop_size, self.bs, self.num_workers, self.dl)
        

class FrameTrackingModel(nn.Module):
    def __init__(self, arch='convnext_tiny', pretrained=True, n_in=3, concat_pool=True, mlp_n=64):
        super(FrameTrackingModel, self).__init__()
        
        model = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=n_in)
        self.body = TimmBody(model, pretrained, n_in=n_in)
        
        self.pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        
        self.mlp = nn.Sequential(
            nn.Linear(18, mlp_n),
            nn.LayerNorm(mlp_n),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.fc = nn.Linear(model.num_features*[1, 2][concat_pool]+mlp_n, 2)
        
    def forward(self, frame, feats):
        # print(frame.shape, feats.shape)
        x1 = self.body(frame)
        x1 = self.pool(x1)
        x1 = self.flatten(x1)
        x2 = self.mlp(feats)
        # print(x1.shape, x2.shape)
        return self.fc(torch.cat([x1, x2], dim=1))