#!/bin/bash

recordings=(
  ES2002a
  ES2002b
  ES2002c
  ES2002d
  ES2003a
  ES2003b
  ES2004a
  ES2004b
  ES2005a
  ES2005b

for rec in "${recordings[@]}"; do
  wget -P "amicorpus/${rec}/audio" \
  "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/${rec}/audio/${rec}.Mix-Headset.wav"
done
wget  https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
unzip ami_public_manual_1.6.2.zip -d ami_public_manual_1.6.2