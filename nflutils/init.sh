#!/bin/bash

mkdir /root/.kaggle
cp kaggle.json /root/.kaggle/
chmod 600 /root/.kaggle/kaggle.json

pip install -r requirements.txt

sudo apt-get update
sudo apt-get install htop