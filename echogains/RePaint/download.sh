#!/bin/bash

(
mkdir -p data/pretrained
cd data/pretrained


)

# data
(
gdown https://drive.google.com/uc?id=1Q_dxuyI41AAmSv9ti3780BwaJQqwvwMv
unzip data.zip
rm data.zip
)