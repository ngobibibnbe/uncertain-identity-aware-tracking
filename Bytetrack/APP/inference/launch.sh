#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=1-00:0:0

conda activate  HMMBytetrack
#cd ..
#python3 tools/demo_track_m.py video -f exps/example/mot/yolox_s_mix_det.py -c /home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sophie/cdpq/ByteTrack/models/yoloX_s_pig_trained_model_400_images.tar --path /home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sophie/cdpq/ByteTrack/videos/GR77_20200512_111314.mp4  --fuse --save_result --device cpu --fps 25 --conf 0.2 --track_thres 0.2  --match_thresh 1.0 --nms 0.45 --tsize 416 
#cd test/inference



#cd /home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sophie/cdpq/ByteTrack
python ATQ.py
python forward_backward.py