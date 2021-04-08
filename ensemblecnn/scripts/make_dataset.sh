#!/usr/bin/env bash

echo ""
echo "-------------------------------------------------"
echo "| Index DFDC dataset                            |"
echo "-------------------------------------------------"
# put your dfdc source directory path and uncomment the following line
# DFDC_SRC=../dfdc/train
# python index_dfdc.py --source $DFDC_SRC --videodataset ./data/dfdc_videos_train.pkl

echo ""
echo "-------------------------------------------------"
echo "| Index FF dataset                              |"
echo "-------------------------------------------------"
# put your ffpp source directory path and uncomment the following line
# FFPP_SRC=/your/ffpp/source/directory
# python index_ffpp.py --source $FFPP_SRC


echo ""
echo "-------------------------------------------------"
echo "| Extract faces from DFDC                        |"
echo "-------------------------------------------------"
# put your source and destination directories and uncomment the following lines
DFDC_SRC=../dfdc/test
VIDEODF_SRC=./data/dfdc_videos_test.pkl
FACES_DST=./data/face_test
FACESDF_DST=./data/face_test_df
CHECKPOINT_DST=./data/test_checkpoint
python extract_faces.py \
--source $DFDC_SRC \
--videodf $VIDEODF_SRC \
--facesfolder $FACES_DST \
--facesdf $FACESDF_DST \
--checkpoint $CHECKPOINT_DST

echo ""
echo "-------------------------------------------------"
echo "| Extract faces from FF                         |"
echo "-------------------------------------------------"
# put your source and destination directories and uncomment the following lines
# FFPP_SRC=/your/dfdc/source/folder
# VIDEODF_SRC=/previously/computed/index/path
# FACES_DST=/faces/output/directory
# FACESDF_DST=/faces/df/output/directory
# CHECKPOINT_DST=/tmp/per/video/outputs
# python extract_faces.py \
# --source $FFPP_SRC \
# --videodf $VIDEODF_SRC \
# --facesfolder $FACES_DST \
# --facesdf $FACESDF_DST \
# --checkpoint $CHECKPOINT_DST