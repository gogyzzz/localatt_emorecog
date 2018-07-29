#!/bin/bash

osmdir=$1
wavcat=$2
lldhtk=$3

conf=$osmdir/config/localatt_lld.conf

wavhtk=wav_htk.list.tmp
wav=wav.list.tmp

awk '{print $1}' $wavcat > $wav
echo ""
echo "<$wav>"
head -2 $wav

paste -d ' ' $wav $lldhtk > $wavhtk
echo ""
echo "<$wavhtk>"
head -2 $wavhtk

cat $wavhtk | parallel --colsep ' ' $osmdir/SMILExtract -C "$conf" -I {1} -O {2}

rm $wav
rm $wavhtk
