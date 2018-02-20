#!/bin/sh

cd data
bash fetch.sh
bash decode.sh
cp -r data/MUS-STEMS-SAMPLE/ data/EST
rm -r data/EST/test
cd ..
