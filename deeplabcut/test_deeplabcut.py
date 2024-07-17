"""
ici on va tester la video complète entrainé sur toutes les données de margo
ensuite la réutiliser pour les mini tracking 
"""
###what next: change config_file to the basic one and test by replacing inference_cfg 
##
from dataclasses import replace
import math
from operator import index
from random import shuffle
import numpy as np
import deeplabcut 
from moviepy.editor import VideoFileClip
import os
import pandas as pd
import glob
from scipy.optimize import linear_sum_assignment
import yaml
os.chdir('/home/sophie/uncertain-identity-aware-tracking/deeplabcut/pig_tracking_dataset')

config_path_center_tail = "/home/sophie/uncertain-identity-aware-tracking/deeplabcut/pig_tracking_dataset/config_center_tail.yaml"#'/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sophie/cdpq/deeplabcut/CDPQ_test-CDPQ_experiment-2022-02-22/config_center_tail.yaml'
config_path_full = "/home/sophie/uncertain-identity-aware-tracking/deeplabcut/pig_tracking_dataset/config_full.yaml"#'/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sophie/cdpq/deeplabcut/CDPQ_test-CDPQ_experiment-2022-02-22/config_center_tail.yaml'
config_path_ears = "/home/sophie/uncertain-identity-aware-tracking/deeplabcut/pig_tracking_dataset/config_ears.yaml"#'/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sophie/cdpq/deeplabcut/CDPQ_test-CDPQ_experiment-2022-02-22/config_center_tail.yaml'

config_path_ears_tail = "/home/sophie/uncertain-identity-aware-tracking/deeplabcut/pig_tracking_dataset/config_ears_tail.yaml"

def test_deeplabcut(config_path, iteration=0):
    shuffle =1
    with open(config_path) as f :
        print(f)
    print("file config opened")
    #deeplabcut.check_labels(config_path)

    #test_graph = [[0,1],[0,2],[1,3],[2,3],[3,4]]  # These are indices in the list of multianimalbodyparts
    #!!!!make sure in  dlc-model/train/inference_cfg.yaml you put minimalnumberofconnections: 0 ###-- ca ne marche pas en tt cas pas dans le create_training 



    ##################uncomment down################
    """deeplabcut.create_multianimaltraining_dataset(config_path)#, paf_graph=my_better_graph)
    deeplabcut.dropannotationfileentriesduetodeletedimages(config_path) #nettoyer les parties non labélisées correctement dans le H5
    
    
    deeplabcut.create_multianimaltraining_dataset(
        config_path,
        Shuffles=[1],
        net_type="dlcrnet_ms5",
    # paf_graph=test_graph, # , paf_graph='config' 
    )
    
    print("starting training")
    deeplabcut.train_network(
        config_path,
        shuffle=1,
        saveiters=10000,
        maxiters=130000,
        allow_growth=False,
        gputouse=2,

    )"""
    print("training finished")

    deeplabcut.evaluate_network(
        config_path,
        Shuffles=[1],
        gputouse=2,
        plotting="individual"
    )
    

    video_path="/home/sophie/uncertain-identity-aware-tracking/deeplabcut/pig_tracking_dataset/evaluation-results/iteration-"+str(iteration)+"/CDPQ_testFeb22-trainset90shuffle1/GR77_20200512_111314.mp4"
    scorername= deeplabcut.analyze_videos(config_path,video_path, videotype='.mp4',shuffle=1,  save_as_csv=True) # n_tracks=15, calibrate=True,gputouse=0,auto_track=False)#auto_track=True#deeplabcut.convert_detections2tracklets(config_path,["/home/sophie/uncertain-identity-aware-tracking/deeplabcut/videos/GR77_20200512_111314.mp4"], videotype='.mp4', window_size=0, calibrate=False, overwrite=True)
    deeplabcut.create_video_with_all_detections(config_path, video_path, videotype='.mp4')
    deeplabcut.convert_detections2tracklets(config_path,[video_path],identity_only=False, videotype='.mp4', overwrite=True, track_method="ellipse")
    deeplabcut.stitch_tracklets(config_path,[video_path], videotype='.mp4',n_tracks=15, track_method="ellipse")
    deeplabcut.create_video_with_all_detections(config_path, [video_path], videotype='.mp4')
    deeplabcut.create_labeled_video(config_path, [video_path], videotype='.mp4', draw_skeleton=True,track_method="ellipse",color_by="individual",displayedindividuals="all",displayedbodyparts=["center","tail","lear","rear","head"])

    

    #dans des sous dossiers couper la video en morceau et appliquer le stitching par morceaux garder un overlap entre les morceaux
    #reconstruire le grand h5file dans le fichier racine  pour faire le create labeled video
    deeplabcut.create_labeled_video(config_path, ['/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sophie/cdpq/deeplabcut/CDPQ_test-CDPQ_experiment-2022-02-22/videos/center_tail/GR77_20200512_111314C.mp4'], videotype='.mp4', draw_skeleton=True,color_by="individual",displayedindividuals="all",displayedbodyparts=["center","tail"], track_method="ellipse")

    
    deeplabcut.create_video_with_all_detections(config_path, [video_path], videotype='.mp4')
    deeplabcut.create_labeled_video(config_path, [video_path], videotype='.mp4', draw_skeleton=True,track_method="ellipse",color_by="individual",displayedindividuals="all",displayedbodyparts=["center","tail"])
    

    # on met le stitch tracklet après parce qu'il peut causer des erreurs si le modèle n'est pas bien entrainé
    deeplabcut.stitch_tracklets(config_path,[video_path], videotype='.mp4',track_method="ellipse")
    deeplabcut.create_video_with_all_detections(config_path, [video_path], videotype='.mp4')
    deeplabcut.create_labeled_video(config_path, [video_path], videotype='.mp4', draw_skeleton=True,track_method="ellipse",color_by="individual",displayedindividuals="all",displayedbodyparts=["center","tail","lear","rear","head"])
    

#test_deeplabcut(config_path=config_path_full)
#test_deeplabcut(config_path=config_path_full, iteration=0)

#test_deeplabcut(config_path=config_path_ears, iteration=2)
#test_deeplabcut(config_path=config_path_center_tail, iteration=1)
test_deeplabcut(config_path=config_path_ears_tail, iteration=3)
















