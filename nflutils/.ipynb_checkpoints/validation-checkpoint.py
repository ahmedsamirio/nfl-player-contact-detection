import cv2

import pandas as pd
import numpy as np

from sklearn.metrics import matthews_corrcoef

from tqdm.notebook import tqdm


def smooth_predictions(val_df, w, center):
    val_df_new = pd.DataFrame()
    for group, group_df in tqdm(val_df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2', 'view'])):
        group_df['contact_pred_rolling'] = group_df.contact_pred.rolling(w, center=center).mean().bfill().ffill().fillna(0)
        # group_df['contact_pred_rolling'] = np.where(group_df.contact_pred_rolling.isna(), group_df.contact_pred, group_df.contact_pred_rolling)
        val_df_new = pd.concat([val_df_new, group_df])
    return val_df_new


def get_matthews_corrcoef(val_dist_agg, pred_col='contact_pred_rolling'):
    out = np.where(val_dist_agg['contact_pred'].isna(),
                   val_dist_agg['distance'] <= 1, 
                   val_dist_agg[pred_col] > 0.5).astype(int)
    
    return matthews_corrcoef(val_dist_agg['contact'], out)


def merge_combo_val(df_combo, val_df, pred_col='contact_pred_rolling'):
    val_dist = df_combo[df_combo.game_play.isin(val_df.game_play.unique())].copy()

    val_dist["distance"] = val_dist["distance"].fillna(99)  # Fill player to ground with 9    
    val_dist_agg = val_dist.merge(val_df.groupby('contact_id', as_index=False)[pred_col].mean(), how='left', on='contact_id')
    
    if pred_col != 'contact_pred':
        val_dist_agg = val_dist_agg.merge(val_df.groupby('contact_id', as_index=False)['contact_pred'].mean(), how='left', on='contact_id')
    
    return val_dist_agg


def find_best_window(df_combo, val_df, w_min, w_max):
    
    # Calculate multiple rolling windows
    val_df_new = pd.DataFrame()
    for group, group_df in tqdm(val_df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2', 'view'])):
        for i in range(w_min, w_max):
            for center in [True, False]:
                group_df[f'contact_pred_rolling_{i}_{center}'] = group_df.contact_pred.rolling(i, center=center).mean()
                val_df_new = pd.concat([val_df_new, group_df])
                
    results = {}
    best_score = 0
    best_config = ''
    
    # Evaluate on the calculated windows
    for i in range(w_min, w_max):
        for center in [True, False]:
            
            val_dist_agg = merge_combo_val(df_combo, val_df_new, f'contact_pred_rolling_{i}_{center}')

            results['Baseline'] = get_matthews_corrcoef(val_dist_agg, 'contact_pred')
            results[f'{i}_{center}'] = get_matthews_corrcoef(val_dist_agg, f'contact_pred_rolling_{i}_{center}')
            
            if results[f'{i}_{center}'] > best_score:
                best_config = f'Window: {i}, Center: {center}'
                best_score = results[f'{i}_{center}']
    
    print('Baseline:', results['Baseline'])
    print('Best config ->', best_config) 
    print('Best conifg score:', best_score)
            
    return results


def rank_game_plays(val_dist_agg, ascending=False):
    game_plays_scores = val_dist_agg.groupby('game_play').apply(get_matthews_corrcoef).sort_values(ascending=ascending).reset_index()
    game_plays_scores.columns = ['game_play', 'score']
    
    return game_plays_scores


def rank_pairs(val_dist_agg, ascending=False):
    cols = ['game_play', 'nfl_player_id_1', 'nfl_player_id_2']
    pairs_contact = val_dist_agg.groupby(cols).agg({'contact': 'sum'}).reset_index()
    pairs_scores = val_dist_agg.groupby(cols).apply(get_matthews_corrcoef).sort_values(ascending=ascending).reset_index()
    pairs_scores.columns = cols + ['score']
    
    return pd.merge(pairs_scores, pairs_contact, on=cols, how='left')


def plot_pair(pair_row, val_dist_agg, ax=None):
    pair_dict = pair_row.to_dict()
    game_play = pair_dict['game_play']
    player_1 = pair_dict['nfl_player_id_1']
    player_2 = pair_dict['nfl_player_id_2']
    score = pair_dict['score']
    contact = pair_dict['contact']
    
    (val_dist_agg.query('game_play == @game_play and nfl_player_id_1 == @player_1 and nfl_player_id_2 == @player_2')
            .groupby('step')
            .contact.first()
            .plot(label='truth', color='r', ax=ax))
    
    (val_dist_agg.query('game_play == @game_play and nfl_player_id_1 == @player_1 and nfl_player_id_2 == @player_2')
            .groupby('step')
            .contact_pred_rolling.apply(lambda x: int(np.mean(x) > 0.5))
            .plot(linestyle='--', linewidth= 1.5, color='b', label='pred', ax=ax))
    
    if ax is None:
        plt.legend()
        plt.title(f'{game_play} - {player_1} - {player_2} - {round(score, 2)} - {contact}')
        
    else:
        ax.legend()
        ax.set_title(f'{game_play} - {player_1} - {player_2} - {round(score, 2)} - {contact}')
        
        
        
def plot_top_pairs(pairs_scores, n, ascending=True):
    pairs_scores = pairs_scores.sort_values('score', ascending=ascending)
    
    fig, axes = plt.subplots((n//3)+((n%3)!=0), 3, figsize=(15, (n//3)*6))
    axes = axes.flatten()
    
    for i, row in pairs_scores.iloc[:n].reset_index().iterrows():
        plot_pair(row, val_dist_agg, axes[i])
        
        
def plot_helmet_on_frame(frame, left, width, top, height, color=(0,0,0), thickness=1):
    startpoint = (int(left), int(top))
    endpoint = (int(left + width), int(top + height))
    return cv2.rectangle(frame, startpoint, endpoint, color, thickness)


def plot_player_name_on_frame(frame, name, left, top, color=(0,0,0), thickness=1):
    return cv2.putText(frame, name,
                       (int(left + 1), max(0, int(top - 20))),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       color,
                       thickness=thickness)


def plot_contact_line(frame, contact_tuple, color=(255,0,0), thickness=2):
    return cv2.line(frame, 
                 (int(contact_tuple.left_1) + int(contact_tuple.width_1 / 2), int(contact_tuple.top_1) + int(contact_tuple.height_1 / 2)),
                 (int(contact_tuple.left_2) + int(contact_tuple.width_2 / 2), int(contact_tuple.top_2) + int(contact_tuple.height_2 / 2)),
                 color=(255, 0, 0),
                 thickness=thickness)


def video_with_pair_contact(pair_df, tr_helmets, gp, frames_path, view='Sideline', verbose=True):
    # Create output video
    VIDEO_CODEC = "MP4V"
    
    fps = 59.94
    width = 1280*2
    height = 720
    
    output_path = f"contact_{gp}_{view}.mp4"
    tmp_output_path = "tmp_" + output_path.split('.')[0] + ".mp4"
    output_video = cv2.VideoWriter(
        tmp_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height)
    )
    
    # Filter gameplay pairs
    gp_pair_df = pair_df.query('game_play == @gp and view == @view')
    gp_helmets_df = tr_helmets.query('game_play == @gp and view == @view')
    
    # Glob gameplay frames
    frames_paths = sorted(glob.glob(frames_path+f'/{gp}_{view}*'))
    
    # Loop over frames
    for fp in tqdm(frames_paths):
        
        # Read frame
        frame = read_img(fp)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make copy for predictions
        frame_pred = frame.copy()
        
        # Get frame number
        frame_no = int(fp[-8:-4])
        
        # Filter frame pairs and helmets
        frame_pair_df = gp_pair_df.query('frame == @frame_no and contact == 1')
        frame_pair_preds_df = gp_pair_df.query('frame == @frame_no and contact_pred_rolling > 0.5')
        frame_helmets_df = gp_helmets_df.query('frame == @frame_no')
        
        # Create set for plotted helmets
        players_set = set()
        
        # Plot helmets if exists
        if len(frame_helmets_df) > 0:
            
            # Loop over helmets
            for helmet_tuple in frame_helmets_df.itertuples():

                # Plot helmet if wasn't plotted before
                if helmet_tuple.player_label not in players_set:

                    frame = plot_helmet_on_frame(frame, helmet_tuple.left, helmet_tuple.width, helmet_tuple.top, helmet_tuple.height, color=(0, 0, 0))
                    frame = plot_player_name_on_frame(frame, helmet_tuple.player_label, helmet_tuple.left, helmet_tuple.top, color=(0, 0, 0))
                    
                    frame_pred = plot_helmet_on_frame(frame_pred, helmet_tuple.left, helmet_tuple.width, helmet_tuple.top, helmet_tuple.height, color=(0, 0, 0))
                    frame_pred = plot_player_name_on_frame(frame_pred, helmet_tuple.player_label, helmet_tuple.left, helmet_tuple.top, color=(0, 0, 0))
        
        # Plot pairs if exits
        if len(frame_pair_df) > 0:
        
            # Loop over pairs
            for contact_tuple in frame_pair_df.itertuples():
                
                # Plot first player helmet and name
                frame = plot_helmet_on_frame(frame, contact_tuple.left_1, contact_tuple.width_1, contact_tuple.top_1, contact_tuple.height_1, color=(0, 255, 0))
                frame = plot_player_name_on_frame(frame, contact_tuple.player_label_1, contact_tuple.left_1, contact_tuple.top_1, color=(0, 255, 0))

                # Plot second player helmet and name
                frame = plot_helmet_on_frame(frame, contact_tuple.left_2, contact_tuple.width_2, contact_tuple.top_2, contact_tuple.height_2, color=(0, 255, 0))
                frame = plot_player_name_on_frame(frame, contact_tuple.player_label_2, contact_tuple.left_2, contact_tuple.top_2, color=(0, 255, 0))

                # Plot line between players
                frame = plot_contact_line(frame, contact_tuple)

                # Add both players to plotted set
                players_set.add(contact_tuple.player_label_1)
                players_set.add(contact_tuple.player_label_2)
                
        # Plot pred pairs if exits
        if len(frame_pair_preds_df) > 0:
        
            # Loop over pairs
            for contact_tuple in frame_pair_preds_df.itertuples():
                
                # Plot first player helmet and name
                frame_pred = plot_helmet_on_frame(frame_pred, contact_tuple.left_1, contact_tuple.width_1, contact_tuple.top_1, contact_tuple.height_1, color=(0, 255, 0))
                frame_pred = plot_player_name_on_frame(frame_pred, contact_tuple.player_label_1, contact_tuple.left_1, contact_tuple.top_1, color=(0, 255, 0))

                # Plot second player helmet and name
                frame_pred = plot_helmet_on_frame(frame_pred, contact_tuple.left_2, contact_tuple.width_2, contact_tuple.top_2, contact_tuple.height_2, color=(0, 255, 0))
                frame_pred = plot_player_name_on_frame(frame_pred, contact_tuple.player_label_2, contact_tuple.left_2, contact_tuple.top_2, color=(0, 255, 0))

                # Plot line between players
                frame_pred = plot_contact_line(frame_pred, contact_tuple)

                # Add both players to plotted set
                players_set.add(contact_tuple.player_label_1)
                players_set.add(contact_tuple.player_label_2)
        
        
        final_frame = np.concatenate([frame, frame_pred], axis=1)
        output_video.write(final_frame)
        
    output_video.release()
    
    # Not all browsers support the codec, we will re-load the file at tmp_output_path
    # and convert to a codec that is more broadly readable using ffmpeg
    if os.path.exists(output_path):
        os.remove(output_path)
        
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            tmp_output_path,
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-hide_banner",
            "-loglevel",
            "error",
            "-vcodec",
            "libx264",
            output_path,
        ]
    )
    
    os.remove(tmp_output_path)

    return output_path


def smooth_predictions(val_df, center, ws):
    val_df_new = pd.DataFrame()
    for group, group_df in tqdm(val_df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2', 'view'])):
        group_df = group_df.sort_values('frame').copy()
        for w in ws: 
            group_df[f'contact_pred_rolling_{w}'] = group_df.contact_pred.rolling(w, center=center).mean().bfill().ffill().fillna(0)
        val_df_new = pd.concat([val_df_new, group_df])
    return val_df_new

def merge_combo_val(df_combo, val_df, pred_col='contact_pred_rolling'):
    val_dist = df_combo[df_combo.game_play.isin(val_df.game_play.unique())].copy()

    val_dist["distance"] = val_dist["distance"].fillna(99)  # Fill player to ground with 9    
    val_dist_agg = val_dist.merge(val_df.groupby('contact_id', as_index=False)[pred_col].mean(), how='left', on='contact_id')
    val_dist_agg = val_dist_agg.merge(val_df.groupby('contact_id', as_index=False)['thresh'].first(), how='left', on='contact_id')
    
    if pred_col != 'contact_pred':
        val_dist_agg = val_dist_agg.merge(val_df.groupby('contact_id', as_index=False)['contact_pred'].mean(), how='left', on='contact_id')
        
    return val_dist_agg

def get_matthews_corrcoef(val_dist_agg, pred_col='contact_pred_rolling', dist=1):
    out = np.where(val_dist_agg['contact_pred'].isna(),
                   val_dist_agg['distance'] <= dist, 
                   val_dist_agg[pred_col] > val_dist_agg['thresh']).astype(int)
    
    return matthews_corrcoef(val_dist_agg['contact'], out)