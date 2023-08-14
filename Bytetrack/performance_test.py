import pandas as pd 
import cv2
import numpy as np

import numpy as np  
import pandas as pd

import numpy as np
from scipy.optimize import linear_sum_assignment
#! pip install mpmath
from mpmath import *
import pandas as pd
import numpy as np 
import datetime as dt
import json 
import copy


def read_data(file):
    #here we will go through detections of deepsort 
    import json
    track={}
    with open(file) as f:
        json_file = json.load(f) 

    for frame, detections in json_file.items():
        frame=int(frame)
        track[frame]={}
        for id, detection in detections.items():
            track[frame][id]={}
            track[frame][id]["rectangle"]= tuple(detection)

    return track



def iou (boxA,boxB):
    boxA=[boxA[0],boxA[1],boxA[0]+boxA[2],boxA[1]+boxA[3]]
    boxB=[boxB[0],boxA[1],boxB[0]+boxB[2],boxB[1]+boxB[3]]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1))
    boxBArea = abs((boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou  #np.linalg.norm(np.array([float(track[0]), float(track[1])+float(track[3])/2])-np.array([600,17.5]))


def precise_accuracy_track(label_track, model_track, basic_tracker=False):
    """cette fonction calcule le f1-score recall, accuracy des model par rapport au background
    dedans, les score des trackers et des hMM based tracker sont calculés différemment car quand le hmm based tracker est seuillé, 
    il y'a des id de track qu'il ne retourne pas dans son fichier de resultat.

    Args:
        label_track (_type_): _description_
        model_track (_type_): _description_
        basic_tracker (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    start=True 
    nbr_frame=0
    acc =0
    rec=0
    if basic_tracker==True:
        matching={}
        for frame_id in label_track.keys():
            for label_atq, label_box in label_track[frame_id].items() : 
                if label_atq not in matching.keys() and label_atq!="observed":
                    matching[label_atq]=None 
                    break
            
    
    for frame_id in label_track.keys():
        if frame_id in model_track.keys() :
            nbr_frame+=1
            if basic_tracker==True:
                if start:
                    for label_atq, label_box in label_track[frame_id].items() : 
                        max_iou=float('-inf')
                        for model_atq, model_box in model_track[frame_id].items(): 
                            #print(model_atq)
                            if  label_atq!="observed" and model_atq!="observed":#fix the problem with the obseved on the label 
                                tmp = iou(model_box["rectangle"], label_box["rectangle"])
                                if tmp>=max_iou:
                                    matching [label_atq]=model_atq
                                    max_iou =tmp
                    start=False
                
                matching_frame={}
                for label_atq, label_box in label_track[frame_id].items() :
                        max_iou=float('-inf')
                        for  model_atq, model_box in model_track[frame_id].items():
                            if  label_atq!="observed" and model_atq!="observed" :#fix the problem with the obseved on the label 
                                tmp = iou(model_box["rectangle"], label_box["rectangle"])
                                if tmp>max_iou:
                                    matching_frame[label_atq]=model_atq
                                    max_iou =tmp
                        if label_atq!="observed" and matching[label_atq]==None and  max_iou!=float('-inf') :  
                            matching[label_atq] = matching_frame[label_atq] 
                
            
                shared_items = {k: matching_frame[k] for k in matching_frame if k in matching and matching[k] == matching_frame[k]}
                #print(len(shared_items), len(matching.keys()))  
                if len(matching.keys())!=0 and len(matching_frame.keys())!=0:
                    acc +=  len(shared_items)/len(matching_frame.keys())
                    rec += len(shared_items)/ len(label_track[frame_id].keys())
                
            else:  #Here is for the HMM approach 
                
                if frame_id==10066:
                    print("stop")
                matching_frame={}
                for model_atq, model_box in model_track[frame_id].items() :
                        max_iou=float('-inf')
                        for  label_atq, label_box in label_track[frame_id].items():
                            if  label_atq!="observed" and model_atq!="observed" :#fix the problem with the obseved on the label 
                                tmp = iou(model_box["rectangle"], label_box["rectangle"])
                                if tmp>max_iou:
                                    matching_frame[model_atq]=label_atq
                                    max_iou =tmp
                filtered ={key:value for key,value in matching_frame.items() if value==key }
                if len(matching_frame.keys())!=0:
                    acc = acc + len(filtered.keys())/ len(matching_frame.keys())
                    rec = rec+ len(filtered.keys())/ len(label_track[frame_id].keys())
                    if len(filtered.keys())/ len(label_track[frame_id].keys())>1:
                        print("stop")


    
    acc = acc/nbr_frame
    rec=rec/nbr_frame
    f1=2*acc*rec/(acc+rec)
    print(acc, rec) 
    return acc  , rec , f1            
               
                            


label_file= "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/labels_with_atq.json"
#bytetrack_result_file = "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314tracking_resut.json"
track_base = "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/tmp"


label_track = read_data(label_file)
#bytetrack_track= read_data(bytetrack_result_file)

import pandas as pd 
from ATQ import adding_atq
from forwardBackward import process_forwad_backward
hmm_result_with_visits=pd.DataFrame(columns=["nbr of visits", "accuracy", "recall", "f1"])


import os 
import time 

def score_for_mot_trackers():
    for filename in os.listdir(track_base):
        track= read_data(track_base+"/"+filename)
        acc, rec, f1= precise_accuracy_track(label_track, track, basic_tracker=True)
        new_row= {'nbr of visits':0, 'accuracy':acc, 'recall':rec, "f1":f1}
        print(filename,new_row)



def score_for_various_artificial_observations():
    for i in range (2, len(label_track.keys()), 100): #[10, 100]:#  [18]: #
        observation_file="videos/GR77_20200512_111314DBN_resut_with_observations_visits_"+str(i)+".json"
        print(i, "ok")
        
        adding_atq(i, output_file=observation_file)
        process_forwad_backward(observation_file,nbr_visit=i, json_save_path="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314_with_atq_tracking_with_HMM_resut"+str(i)+".json")
        Hmm_result_file="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314_with_atq_tracking_with_HMM_resut"+str(i)+".json"

        hmm_track = read_data(Hmm_result_file)
        acc, rec, f1= precise_accuracy_track(label_track, hmm_track)
        new_row= {'nbr of visits':i, 'accuracy':acc, 'recall':rec, "f1":f1}
        print(new_row)

        hmm_result_with_visits = pd.concat([hmm_result_with_visits, pd.DataFrame([new_row])], ignore_index=True)
        

    #hmm_result_with_visits.to_csv('accuracy_over_nbr_of_visits_with_track_helping.csv')


def score_for_visit_at_feeder():
    print("ok")
    observation_file="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314DBN_resut_with_observations_feeder.json"
    #adding_atq(1, output_file=observation_file, feeder=True)
    #process_forwad_backward(observation_file,nbr_visit=1, json_save_path="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314_with_atq_tracking_with_HMM_result_feeder.json")
    Hmm_result_file="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314_with_atq_tracking_with_HMM_result_feeder.json"
    hmm_track = read_data(Hmm_result_file)
    acc, rec, f1= precise_accuracy_track(label_track, hmm_track, basic_tracker=False)
    new_row= {'nbr of visits':"feeder", 'accuracy':acc, 'recall':rec, "f1":f1}
    print(new_row)
    #hmm_result_with_visits = pd.concat([hmm_result_with_visits, pd.DataFrame([new_row])], ignore_index=True)    
    #hmm_result_with_visits.to_csv('accuracy_over_nbr_of_visits_with_track_helping.csv')
    
#process_forwad_backward()


import argparse
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Export a Conda environment with all its packages.")
    parser.add_argument("mode", type=str, help="Name of the mode.")
    args = parser.parse_args()
    if args.mode == "feeder":
        score_for_visit_at_feeder()
    if args.mode == "artificial_visits":
        score_for_various_artificial_observations()
    if args.mode == "tracker_test":
        score_for_mot_trackers()