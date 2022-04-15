#!/bin/bash

#$ -cwd
#$ -l m_mem_free=54G
#$ -l gpu=1

# run program
python train_invivo.py --name $1 --regularizer $2 --activation $3 $4 --dropout1 0.2 --batch 128