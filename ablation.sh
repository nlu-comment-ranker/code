#!/bin/bash

HERE=$(dirname $0)
DATA=$1
TARGET="log_score"
MODEL="rf"
OUTDIR=$2
mkdir -p $OUTDIR

python $HERE/classifier.py $DATA -t $TARGET -c $MODEL -s $OUTDIR/all_abl --fg all

python $HERE/classifier.py $DATA -t $TARGET -c $MODEL -s $OUTDIR/combo_abl --fg combo

python $HERE/classifier.py $DATA -t $TARGET -c $MODEL -s $OUTDIR/all_text_abl --fg all_text

python $HERE/classifier.py $DATA -t $TARGET -c $MODEL -s $OUTDIR/user_abl --fg user_only

python $HERE/classifier.py $DATA -t $TARGET -c $MODEL -s $OUTDIR/base_meta_ctxt_abl --fg baseline metadata context

python $HERE/classifier.py $DATA -t $TARGET -c $MODEL -s $OUTDIR/base_meta_abl --fg baseline metadata

python $HERE/classifier.py $DATA -t $TARGET -c $MODEL -s $OUTDIR/base_abl --fg baseline

python $HERE/classifier.py $DATA -t $TARGET -c $MODEL -s $OUTDIR/dummy_abl -f position_rank
