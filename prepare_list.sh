#!/bin/bash

wavcat=$1
lldhtk=$2
utt=$3
htkdir=$4

mkdir -p $htkdir

echo ""
echo "< $wavcat >"
head -2 $wavcat

awk '{print $1}' $wavcat | awk -F/ '{print $NF}' | sed 's/\.wav//g' > $utt

echo ""
echo "< $utt >"
head -2 $utt

sed "s|$|.htk|g; s|^|$htkdir/|g" $utt > $lldhtk

echo ""
echo "< $lldhtk >"
head -2 $lldhtk

echo ""
echo "done."

rm $utt




