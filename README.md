# NFL 1st and Future - Player Contact Detection

This repo contains my solution for the NFL 1st and Future - Player Contact Detection Kaggle competition, which yields the 35th position (out of 939 teams) and a silver medal.

<img src="https://github.com/ahmedsamirio/nfl-player-contact-detection/blob/main/data/output/ezgif.com-optimize (1).gif" width="1280" height="280"/>

The pipeline for contatcct deteciton is decoupled into two separate pipelines:
1. A object detection and tracking pipeline for tracking players
2. A contact detection pipeline for detecting player-player contact and player-ground contact


This repo includes the second part of the pipeline which depends on the output of the first part consisting of
1. Player helmets bounding boxes (the output of the first pipeline)
2. Player tracking data (recorded using sensors)

