#!/bin/bash
#
# Data spliting (spliting ratio= trainig data/total)
#

IG_IN='/home/sada/ML/SAKIGAKE2/training/IG_mol_gpu.txt'
SDF='/home/sada/ML/SAKIGAKE2/training/input/HF_test_SOLV.sdf'
IG_out='MAP_IG_features.txt'
#
# 
#
python hansch.py --IG_in ${IG_IN} --IG_out ${IG_out} --sdf ${SDF}
