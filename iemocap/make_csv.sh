#!/bin/bash
if [ $# -ne 4 ]; then
  echo "Usage: $0 iemocap/utt.list iemocap/wav_cat.list iemocap/ iemocap/full_dataset.csv"
  exit 1;
fi

utt=$1
wavcat=$2
iemocap_dir=$3
csv=$4

spk="$iemocap_dir/spk.list"
sess="$iemocap_dir/sess.list"
gender="$iemocap_dir/gender.list"
emo="$iemocap_dir/cat.list"

gawk '{print $2}' $wavcat > $emo

cut -c4-6 $utt > $spk

cut -c6 $utt > $gender

cut -c4-5 $utt > $sess

# check
echo "<$emo>"; head -2 $emo
echo "<$spk>"; head -2 $spk
echo "<$gender>"; head -2 $gender
echo "<$sess>"; head -2 $sess

header="utterance,speaker,gender,session,emotion"

cat <(echo $header) <(paste -d ',' $utt $spk $gender $sess $emo) > $csv

echo "<$csv>"
head -3 $csv
