#!/bin/bash

##
# Flair cross-domain experiment script generator
# generates a script with static commands
# to run pairwise cross-domain experiments
# on individual flair files.
#
# Generates a file "flairx.sh" that will run the
# desired series of experiments.
##

FLAIRS=(astronomy biology chemistry computing earth_sciences engineering mathematics medicine neuroscience physics psychology)
# FLAIRS=(astronomy biology)

FG=${1:-"all"}
SCORE=${2:-"score"}
CLF=${3:-"rf"}

OUTFILE="flairx.sh"
echo '#!/bin/bash' > $OUTFILE
echo 'trap exit ERR' >> $OUTFILE
echo 'set -x' >> $OUTFILE
echo 'HERE=$(dirname $0)' >> $OUTFILE
echo 'DFBASE=${1:-"data/data-askscience-feb18.ALL"}' >> $OUTFILE
echo 'OUTDIR=${2:-"tmp/flairx"}' >> $OUTFILE
echo 'mkdir -p $OUTDIR' >> $OUTFILE
chmod +x $OUTFILE

# Pairwise
for flair1 in ${FLAIRS[@]}
do
	echo "" >> $OUTFILE
	for flair2 in ${FLAIRS[@]}
	do
		if [ $flair1 != $flair2 ]
		then
			echo $flair1 "->" $flair2
			CMD='stdbuf -oL -eL $HERE/classifier.py $DFBASE'".$flair1 --crossdomain "'$DFBASE'".$flair2 --fg $FG -t $SCORE -c $CLF -s "'$OUTDIR'"/$flair1-on-$flair2 | tee "'$OUTDIR'"/$flair1-on-$flair2.log"
			echo -e $CMD >> $OUTFILE
		fi
	done
done