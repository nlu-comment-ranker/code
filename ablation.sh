#!/bin/bash

DATA=$1
TARGET="score"
MODEL="rf"

python classifier.py $DATA -t $TARGET -c $MODEL -s all_abl --fg all

python classifier.py $DATA -t $TARGET -c $MODEL -s combo_abl --fg combo

python classifier.py $DATA -t $TARGET -c $MODEL -s all_text_abl --fg all_text

python classifier.py $DATA -t $TARGET -c $MODEL -s user_abl --fg user_only

python classifier.py $DATA -t $TARGET -c $MODEL -s base_meta_ctxt_abl --fg baseline metadata context

python classifier.py $DATA -t $TARGET -c $MODEL -s base_meta_abl --fg baseline metadata

python classifier.py $DATA -t $TARGET -c $MODEL -s base_abl --fg baseline

python classifier.py $DATA -t $TARGET -c $MODEL -s dummy_abl -f position_rank
