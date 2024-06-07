#!/bin/bash
#
# CLUSTERING: Kmeans or PIC
#
ACCELERATOR='cpu'
DEVICES=1
CONFIG_FILE="./config/regression.py"
#
# prepare ckpt working directory
#
CKPT_DIR=/scratch/$USER/scr_dir
if [ ! -d ${CKPT_DIR} ]; then
   mkdir ${CKPT_DIR}
fi
#
# check whether this calculation is hyper-parameter fitting or not.
#
if [[ $1 = "-hyper" ]]; then
  python training_optuna.py -config ${CONFIG_FILE} -ckpt_scr ${CKPT_DIR} -accelerator ${ACCELERATOR} -devices ${DEVICES} -hyper
else
  python training_optuna.py -config ${CONFIG_FILE} -ckpt_scr ${CKPT_DIR} -accelerator ${ACCELERATOR} -devices ${DEVICES}
fi
#
# remove last.ckpt
#
rm -f last.ckpt
#
# remove ckpt working directory
#
rm -rf ${CKPT_DIR}
