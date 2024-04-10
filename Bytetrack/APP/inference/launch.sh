#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=1-00:0:0

conda activate  HMMBytetrack



#cd /ByteTrack
python ATQ.py
python forward_backward.py