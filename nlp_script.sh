#!/bin/bash

# Change to submission directory from which we submitted the job.
#$ cd /atlas/u/jdunnmon/tutorial/cnn-text-classification-tf
#$ -S /bin/bash

# mail this address
#$ -M $jdunnmon@cs.stanford.edu
# send mail on begin, end, suspend
#$ -m bes

# request 2GB of RAM, not hard-enforced on FarmShare
#$ -l mem_free=2G

# request 6 mins of runtime, is hard-enforced on FarmShare
# -l h_rt=04:00:00

# Perform tasks
source ~/.bashrc.user
cd /atlas/u/jdunnmon/tutorial/cnn-text-classification-tf
python train_nlp.py > out_gop.txt
