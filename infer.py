from nflutils.dataprep import *
from nflutils.training import *
from nflutils.validation import *

from pathlib import Path
from tqdm import tqdm

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inference script for player contact detection model')
    parser.add_argument('--name', help='Experiment name', type=str)
    parser.add_argument('--dist_thresh', help='Minimum distance between players', type=int, default=2)
    parser.add_argument('--sample_n', help='Sample every n frames', type=int, default=2)
    parser.add_argument('--visualize', help='Output video for results', type='store_true')

    args = parser.parse_args()

    # Accessing the argument values
    name = args.name
    dist_thresh = args.dist_thresh
    sample_n = args.sample_n
    BASE_DIR = Path("data")
    frames_path = 'frames/content/work/frames/train'
    utils_path = 'nflutils'


    BASE_DIR = Path('data')


    # Player tracking data
    te_tracking = pd.read_csv(
        BASE_DIR/"raw/test_player_tracking.csv", parse_dates=["datetime"]
    )

    # Baseline helmet detection labels
    te_helmets = pd.read_csv(BASE_DIR/"raw/test_baseline_helmets.csv")

    te_video_metadata = pd.read_csv(
        BASE_DIR/"raw/test_video_metadata.csv",
        parse_dates=["start_time", "end_time", "snap_time"],
    )

    ss = expand_contact_id(ss)
    ss_dist = create_features(ss, te_tracking, merge_col='step')

    ss_combo = ss_dist.copy()
    ss_combo = ss_combo[(ss_combo.distance.isna()) | (ss_combo.distance <= dist_thresh)]



    ss_dist_with_helmets = pd.DataFrame()
    for gp in tqdm(ss_dist.game_play.unique()):
        ss_dist_with_helmets = pd.concat([ss_dist_with_helmets,
                                            merge_tracking_and_helmets_ts(ss_combo, te_helmets, te_video_metadata, gp, 'Sideline'),
                                            merge_tracking_and_helmets_ts(ss_combo, te_helmets, te_video_metadata, gp, 'Endzone')])

    ss_dist_with_helmets = calc_two_players_helmets_center(ss_dist_with_helmets)
    ss_dist_with_helmets = ss_dist_with_helmets.query("view != 'Endzone2'")
    ss_dist_with_helmets = ss_dist_with_helmets[~ss_dist_with_helmets.view.isna()]
    ss_dist_with_helmets = ss_dist_with_helmets[(ss_dist_with_helmets.left_1.notnull()) & (ss_dist_with_helmets.frame.notnull())]
    ss_dist_with_helmets = ss_dist_with_helmets[(ss_dist_with_helmets.left_2.notnull()) | (ss_dist_with_helmets.nfl_player_id_2 == "G")]
    ss_dist_with_helmets = ss_dist_with_helmets.astype({'frame': 'int'})

    ss_dist_with_helmets = ss_dist_with_helmets.query('(290 - frame) % @sample_n == 0')

    ss_dist_with_helmets['G_flag'] = np.where(ss_dist_with_helmets.nfl_player_id_2 == 'G', 1, 0)

    ss_dist_with_helmets_g = ss_dist_with_helmets[ss_dist_with_helmets.nfl_player_id_2 == 'G'].copy()
    ss_dist_with_helmets_p = ss_dist_with_helmets[ss_dist_with_helmets.nfl_player_id_2 != 'G'].copy()


    seed = 42
    bs = 64
    channels = 3
    
    means = [0.485, 0.456, 0.406]*2
    stds = [0.229, 0.224, 0.225]*2


    val_transform = A.Compose(
        [
            A.Normalize(mean=means[:channels], std=stds[:channels]),
            ToTensorV2(),
        ]
    )

    test_ds = NFLFrameTrackingDataset(ss_dist_with_helmets_p, transform=val_transform, crop_size=256)

    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True,
    )


    mini_ds = NFLFrameTrackingDataset(ss_dist_with_helmets_p.iloc[:bs], transform=val_transform, crop_size=256)

    mini_loader = DataLoader(
        mini_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True,
    )

    data = DataLoaders(test_loader, test_loader)

    model = torch.load(CFG.name)

    learn = Learner(data, model, CrossEntropyLossFlat(), metrics=[accuracy, MatthewsCorrCoef(), Recall(), Precision(), F1Score()],
                ).to_fp16()


    learn.model = learn.model.cuda()

    preds, _ = learn.get_preds(dl=test_loader)

    ss_dist_with_helmets_p.loc[:, 'contact_pred'] = preds[:, 1].cpu().detach().numpy()

    test_ds = NFLFrameTrackingDataset(ss_dist_with_helmets_g, transform=val_transform, crop_size=256)

    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True,
    )


    mini_ds = NFLFrameTrackingDataset(ss_dist_with_helmets_g.iloc[:bs], transform=val_transform, crop_size=256)

    mini_loader = DataLoader(
        mini_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True,
    )

    data = DataLoaders(test_loader, test_loader)

    model = torch.load(name)

    learn = Learner(data, model, CrossEntropyLossFlat(), metrics=[accuracy, MatthewsCorrCoef(), Recall(), Precision(), F1Score()],
                ).to_fp16()


    learn.model = learn.model.cuda()

    preds, _ = learn.get_preds(dl=test_loader)

    ss_dist_with_helmets_g.loc[:, 'contact_pred'] = preds[:, 1].cpu().detach().numpy()

    w_g, w_p = 60, 18
    thresh_g, thresh_p = 0.43, 0.7

    ss_dist_with_helmets_g = smooth_predictions(ss_dist_with_helmets_g, True, [CFG.w_g])
    ss_dist_with_helmets_p = smooth_predictions(ss_dist_with_helmets_p, True, [CFG.w_p])

    ss_dist_with_helmets_g['contact_pred_rolling'] = ss_dist_with_helmets_g[f'contact_pred_rolling_{CFG.w_g}']
    ss_dist_with_helmets_g['thresh'] = thresh_g
                                                                            
    ss_dist_with_helmets_p['contact_pred_rolling'] = ss_dist_with_helmets_p[f'contact_pred_rolling_{CFG.w_p}']
    ss_dist_with_helmets_p['thresh'] = thresh_p

    cols = ['contact_id', 'distance', 'thresh', 'contact_pred_rolling', 'game_play', 'contact_pred']
    tmp = pd.concat([ss_dist_with_helmets_g[cols], ss_dist_with_helmets_p[cols]])

    video_with_pair_contact(tmp, te_helmets_df, frames_path)
