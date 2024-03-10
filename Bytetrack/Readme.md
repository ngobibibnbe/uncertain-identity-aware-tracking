# Requirements
pip install -r Bytetrack_requirements.txt
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
pip install -r requirements.txt

# Tracker matching 
We used Bytetrack as tracker, to get the tracker matching matrix 
type those commands:

cd path_to_bytetrack/ByteTrack
python3 tools/demo_track_m.py video -f exps/example/mot/yolox_s_mix_det.py -c path_to_bytetrack/ByteTrack/models/yoloX_s_pig_trained_model_400_images.tar --path path_to_bytetrack/ByteTrack/videos/GR77_20200512_111314.mp4  --fuse --save_result --device cpu --fps 25 --conf 0.2 --track_thres 0.2  --match_thresh 1.0 --nms 0.45 --tsize 416 

# you can reproduce 
our test with artificial visit using the command 
performance_test.py artificial_visits

# To reproduce the test with real feeder data provided by the feeder 
performance_test.py feeder

# To benchmark the models we tested previously you can do 
performance_test.py tracker_test

# To visualize MOTA, IDF1, number of switches etc. you can launch the notebook: Bytetrack/MOT_metric_evaluation/tracking_evaluation.ipynb

#details of further analysis are provided on our github 

Bytetrack: {'nbr of visits': 0, 'accuracy': 0.5192547300415271, 'recall': 0.5053068758652532, 'f1': 0.5121858635007496}


etude des visites: 21 sont rewarded dont 18 bonnes et 3 mauvaises  pkw 21 vu que le feeder a un problème on prend une marge de 2 secondes au debut et à la fin de la visite pour donner plus de chance au modèle d'avoir le bon animal 
Bytetrack: 0.54
re-id : 0.541
h 0.5  {'nbr of visits': 'feeder', 'accuracy': 0.7106643673988, 'recall': 0.5433608640406608, 'f1': 0.6158523688865077}
hungarian 0.3 {'nbr of visits': 'feeder', 'accuracy': 0.6962021742199619, 'recall': 0.5964986587604136, 'f1': 0.6425054468184063}
hungarian 0.2 {'nbr of visits': 'feeder', 'accuracy': 0.6961986446420995, 'recall': 0.5984752223634068, 'f1': 0.6436487972449526}
hungarian 0.1 {'nbr of visits': 'feeder', 'accuracy': 0.6961986446420995, 'recall': 0.5984752223634068, 'f1': 0.6436487972449526}
hungarian 0.0 {'nbr of visits': 'feeder', 'accuracy': 0.59, 'recall': 0.5984752223634068, 'f1': 0.59}
n 0.0 {'nbr of visits': 'feeder', 'accuracy': 0.6796857667251558, 'recall': 0.6036989975998894, 'f1': 0.6394428661784626}
n 0.1 {'nbr of visits': 'feeder', 'accuracy': 0.68058580908009, 'recall': 0.604404913172387, 'f1': 0.6402371623700921}
n 0.2 {'nbr of visits': 'feeder', 'accuracy': 0.6811177668865078, 'recall': 0.604687279401386, 'f1': 0.6406309426138821}
n 0.5  {'nbr of visits': 'feeder', 'accuracy': 0.7016321776486948, 'recall': 0.5476493011435832, 'f1': 0.6151509940267806}


en prenant une 1 seconde de marge à gauche et à droite, pour le best{'nbr of visits': 'feeder', 'accuracy': 0.6696168895342971, 'recall': 0.5470845686855833, 'f1': 0.6021806988403379} donc on garde nos 21 secondes. 


pkw le re_id ne s'ameliore pas vraiment, expliquer ca 

je prend seulement 21 frames de visites, essayer d'enlever la contraintes sur les 21 et laisser toutes les observations (since we corrected labeling errors )




corrected annotations:
0.0
0.07
    H_{'nbr of visits': 0, 'accuracy': 0.5918149089368945, 'recall': 0.4714104193138501, 'f1': 0.5247950871093896}
    n_{'nbr of visits': 'feeder', 'accuracy': 0.5921456808051503, 'recall': 0.4852463645348049, 'f1': 0.5333927238993514}
    n_less: {'nbr of visits': 'feeder', 'accuracy': 0.5721748250337841, 'recall': 0.5012000564732473, 'f1': 0.534340908400742}
0.1
0.2
0.3
0.5
