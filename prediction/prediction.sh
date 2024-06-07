#!/bin/bash
#
# CLUSTERING: Kmeans or PIC
#
ACCELERATOR='cpu'
CONFIG_FILE="./config/regression.py"

if [ -f pred_results.csv ]; then
   rm -f pred_results.csv
fi

python prediction.py -config ${CONFIG_FILE} -accelerator ${ACCELERATOR} 
