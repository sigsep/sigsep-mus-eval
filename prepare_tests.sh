#!/bin/sh

cd data
bash fetch.sh
bash decode.sh
cp -r MUS-STEMS-SAMPLE/ EST
rm -r EST/test
cd ..
