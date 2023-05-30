# NFL 1st and Future - Player Contact Detection

This repo contains my solution for the NFL 1st and Future - Player Contact Detection Kaggle competition, which yields the 35th position (out of 939 teams) and a silver medal.

<img src="https://github.com/ahmedsamirio/nfl-player-contact-detection/blob/main/data/output/ezgif.com-optimize (1).gif" width="1280" height="280"/>

## The pipeline for contact deteciton is decoupled into two separate pipelines:
1. A object detection and tracking pipeline for tracking players
2. A contact detection pipeline for detecting player-player contact and player-ground contact


## This repo includes the second part of the pipeline which depends on the output of the first part consisting of
1. Player helmets bounding boxes (the output of the first pipeline)
2. Player tracking data (recorded using sensors)

## Solution Summary
In this competition, I approached the problem as a binary classification task, where the objective was to predict whether contact occurred between player pairs or with the ground at each timestamp. I used a combination of deep learning techniques and feature engineering to create an effective solution.

## Data Preprocessing
1. I started by analyzing the provided datasets, including the videos, player tracking data, and baseline predictions.
2. To synchronize the videos and player tracking data, I utilized the timestamps provided in the video metadata file.
3. I created Reigons of interest between players using a distance threshold between their sensor tracking data and their helmet predictions

## Model Architecture
1. Two models were trained, one for player-player and another for player-ground contacts
2. The input for each model was a crop of the reigon of interest, overlaid with the bounding boxes of the players where are interested in predicting the contact for
3. I used a pretrained `convnext`, while adding the sensor data as input to the fully connected layer in the end of the base model.
4. The outputs of the model for all frames were smoothed using a sliding window approach, and a distinct threshold was used for each contact type
5. 5 models were trained for each contact type using a 5-fold split, and their outputs were ensembled into the final predictions

## The repository is organized as follows:

`nflutils/`: Contains a utility package created for data exploration, preprocessing, model development, and evaluation steps
`data/`: Contains the necessary data files for training and testing.
`notebooks/`: Jupyter notebooks containing the data exploration, preprocessing, model development, and evaluation steps.
`models/`: Saved model checkpoints and trained weights for the final models.
`README.md`: This file, providing an overview of the competition, solution, and repository structure.
