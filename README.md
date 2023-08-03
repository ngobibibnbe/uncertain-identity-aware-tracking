"# uncertain-identity-aware-tracking" 


# Requirements
pip install -r Bytetrack_requirements.txt
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
pip install -r requirements.txt

# Tracker matching 
We used Bytetrack as tracker, to get the tracker matching matrix 
type those commands:

cd path_to_bytetrack/ByteTrack
python3 tools/demo_track_m.py video -f exps/example/mot/yolox_s_mix_det.py -c path_to_bytetrack/ByteTrack/models/yoloX_s_pig_trained_model_400_images.tar --path path_to_bytetrack/ByteTrack/videos/GR77_20200512_111314.mp4  --fuse --save_result --device cpu --fps 25 --conf 0.2 --track_thres 0.2  --match_thresh 1.0 --nms 0.45 --tsize 416 

# you can reproduce our test with artificial visit using the command 
performance_test.py artificial_visits

# To reproduce the test with real feeder data provided by the feeder 
performance_test.py feeder

# To benchmark the models we tested previously you can do 
performance_test.py tracker_test

# To visualize MOTA, IDF1, number of switches etc. you can launch the notebook: Bytetrack/MOT_metric_evaluation/tracking_evaluation.ipynb

#details of further analysis are provided on our github 




