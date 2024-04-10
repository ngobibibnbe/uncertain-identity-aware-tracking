import pandas as pd
import numpy as np 
import datetime as dt
import json 
import copy
import cv2
import math
import random
random.seed(42)

from datetime import timedelta

###########reading important files and setting the max number of frame ##########
home_folder= " Bytetrack"
video_prev_name="GR77_20200512_111314"




def adding_atq(nbr_visit, output_file, labels_file=home_folder+"/videos/labels_with_atq.json"):
    track_file=home_folder+"/videos/"+video_prev_name+"tracking_resut.json"
    water_file=home_folder+"/videos/eau_parc6.xlsx"
    dbn_file= home_folder+"/videos/"+video_prev_name+"DBN_resut.json"
    feeder_file=home_folder+"/videos/donnees_insentec_lot77_parc6.xlsx"

    with open(track_file) as f:
            tracks = json.load(f) 
    with open(dbn_file) as f:
            dbn_infos = json.load(f) 
    

    max_frame=max([int(i) for i in list(dbn_infos.keys())])
        
    """ add atq depending on the labels file provided, and the number of observations we would like to have 
    
    Returns:
        write in a file:  Bytetrack/videos/"+video_prev_name+"DBN_resut_with_observations.json 
    """
    
    with open(labels_file) as f:
            labels = json.load(f) 
    #V=50#Nomber of random visits 
    idx_selection = [ i for i in range(0, len(labels.keys()), int(len(list(labels.keys()))/nbr_visit) ) ] #random.sample(list(labels.keys()), nbr_visit)
    random_selection = [list(labels.keys())[i] for i  in idx_selection ]
    #########################################################################


    ########################defining utils functions#####################
    def convert_to_json(o):
        try:
            o=o.__dict__
        except:
            o=str(o)
        return o

    def iou (boxA,boxB=[580, 0, 90, 115+20]):
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

    def eucledian_distance(a,b):
        a=np.array(a)
        b=np.array(b)
        dist = np.linalg.norm(a-b)  
        return dist
    
    ##############################################################################""###  


    ########################Here we create the observations (HMM emission matrix) on visit at the feeders #############
    ##############################################################################################
    
    for key in dbn_infos.keys() :
        dbn_infos[key]["observation"]={}
    """tracks_atq={}
    water_visits=pd.read_excel(water_file)
    feeder_visits=pd.read_excel(feeder_file)
    feeder_visits["debut"] = feeder_visits["Date_fin"].combine(feeder_visits["Tfin"], lambda d, t: pd.datetime.combine(d, t))
    feeder_visits['debut'] = [feeder_visits['debut'][idx] - timedelta(seconds=feeder_visits['Duree_s'][idx]) for idx in feeder_visits.index ]

    water_center=[625,70]
    feeder_center=[131, 102]

    #on selectionne les visites qui sont sensées être dans la vidéo
    water_visits = water_visits.loc[(water_visits["debut"]>dt.datetime(2020, 5, 12, 9, 0,0)) & (water_visits["debut"]<dt.datetime(2020, 5, 12, 9, 9,59)) ]
    feeder_visits = feeder_visits.loc[(feeder_visits["debut"]>dt.datetime(2020, 5, 12, 9, 0,0)) & (feeder_visits["debut"]<dt.datetime(2020, 5, 12, 9, 9,59)) ]
    #print(len(water_visits), len(feeder_visits))
    """
    
    
    """
    def add_observations(visits, feeder_center=[625,70]):
        nbr_of_visits=0
        for idx, visit in visits.iterrows(): 
            atq = visit["animal_num"]
            debut = visit['debut']
            fin = visit['fin']
            #on retrouve la frame de chaque visite 
            # de plus on rajoute une marge entre les visites pour éviter les problèmes de confusions 
            #d'identités quand deux animaux viennent bagarer à la mangeoire
            
            # je rajoute +50 frame de marge entre les debuts et fin de visites 
            frame_id_debut = int((debut-dt.datetime(2020, 5, 12, 9, 0,0)).total_seconds()*24.63666666666)+50
            frame_id_fin =  int((fin-dt.datetime(2020, 5, 12, 9, 0,0)).total_seconds()*24.63666666666)-150
            frame_id=frame_id_debut+1
            flag=False
            while frame_id<frame_id_fin: 
                flag=True
                frame_id=frame_id+1
                if max_frame> frame_id:
                    frame=tracks[str(frame_id)]
                    max_d = 0
                    id_track_min =None          
                    for track_id, track in frame.items(): 
                        # on calcule l'iou de chaque animal par rapport à la mangeoire et on vérifie qu'on a au moins un animal à la mangeoire
                        if iou(track)>max_d:
                            id_track_min=track_id
                            max_d =iou(track)
                    if max_d >0 :#and max_d_x<= 620:
                        observation=[]
                        for track in dbn_infos[str(frame_id)]["current"]:
                            ### on  pourrait faire la gaussienne ici
                            tests= [[track["location"][0]+track["location"][2], track["location"][1]+track["location"][3]],[track["location"][0]+track["location"][2], track["location"][1]], [track["location"][0], track["location"][1]+track["location"][3]], [track["location"][0], track["location"][1]]]
                            #track_coin = [track["location"][0]+track["location"][2]/2, track["location"][1]+track["location"][3]/2]
                            min_dist=float('inf')
                            for coin in tests:
                                if eucledian_distance(feeder_center, coin)<min_dist:
                                    min_dist=eucledian_distance(feeder_center, coin)
                                    track_coin=coin
                                    
                                    
                            observation.append(math.exp(eucledian_distance(feeder_center, track_coin)/10))
                            #l'observation est donné par la softmax sur les distance
                        ####transforming distances to probabilities  ****sophie peut être remplacer par une gaussienne plus tard
                        observation = np.array(observation)
                        observation = 1/(1+observation)
                        observation = observation/sum(observation)
                        if max(observation)>=0.5:
                            dbn_infos[str(frame_id)]["observation"][atq]=observation
                            #dbn_infos[str(frame_id)]["observed"]=atq
                else:
                    #print("**not in the video",frame_id)
            
            if flag==True:
                nbr_of_visits+=1
        #print("nbr of rewarded visits",nbr_of_visits)"""
    #add_observations(feeder_visits, feeder_center)
    #add_observations(water_visits, water_center)

    #####################################################################################################
    #############We create the observations using real labels #######

    
    for frame_id in random_selection:
        visitor_id = random.sample(list(labels[frame_id].keys()), 1)[0]
        #visitor_id = list(labels[frame_id].keys())[0]
        #print("***", frame_id, visitor_id)
        if visitor_id!="observed": # and float(visitor_id)>15:  #####twick to modify later ..........................;;;;
            #print(frame_id,labels[frame_id], visitor_id)
            visitor_coordinate=labels[frame_id][visitor_id]
            
            
            frame=tracks[str(frame_id)]
            max_d = 0
            id_track_min =None          
            for  idx, track in enumerate(dbn_infos[str(frame_id)]["current"]): 
                track_id = track['id_in_frame']
                track=track["location"]
                # on calcule l'iou de chaque animal par rapport à la mangeoire et on vérifie qu'on a au moins un animal à la mangeoire
                if iou(track,visitor_coordinate)>max_d:
                    idx_min=idx
                    max_d =iou(track, visitor_coordinate)
            if max_d >0 :#and max_d_x<= 620:
                visitor_coordinate = dbn_infos[str(frame_id)]["current"][idx_min]["location"]
                feeder_center = [visitor_coordinate[0], visitor_coordinate[1]]
                ##print(max_d, "*****")
                observation=[]
                
                for track in dbn_infos[str(frame_id)]["current"]:
                    ### on  pourrait faire la gaussienne ici
                    tests= [ [track["location"][0]+track["location"][2], track["location"][1]+track["location"][3]],[track["location"][0]+track["location"][2], track["location"][1]], [track["location"][0], track["location"][1]+track["location"][3]], [track["location"][0], track["location"][1]] ]
                    #track_coin = [track["location"][0]+track["location"][2]/2, track["location"][1]+track["location"][3]/2]
                    min_dist=float('inf')
                    for coin in tests:
                        if eucledian_distance(feeder_center, coin)<min_dist:
                            min_dist=eucledian_distance(feeder_center, coin)
                            track_coin=coin
                    observation.append(math.exp(eucledian_distance(feeder_center, track_coin)/10))
                    #l'observation est donné par la softmax sur les distance
                ####transforming distances to probabilities  ****sophie peut être remplacer par une gaussienne plus tard
                observation = np.array(observation)
                observation = 1/(1+observation)
                observation = observation/sum(observation)
                if max(observation)>=0.5:
                    dbn_infos[str(frame_id)]["observation"][visitor_id]=observation
                    #dbn_infos[str(frame_id)]["observed"]=atq
            else:
                print("**not in the video",frame_id)

        #if flag==True:
        #    nbr_of_visits+=1



    print("we finished adding observations")
    #exit(0)
    #####################################################################################################

    for frame_id,frame in dbn_infos.items():
        current_to_stay=[]
        ids_to_stay=[]
        ids_to_stay_prev=[]
        if int(frame_id)<max_frame and frame_id!="0" and frame_id!=list(dbn_infos.keys())[-1]: ###???change t300 to -1   int(frame_id)<300 and 
            for idx,i in  enumerate(dbn_infos[str(frame_id)]["current"]):
                for id_prev,j in enumerate(dbn_infos[str(int(frame_id)+1)]["previous"]):
                    if int(i["id_in_frame"])==int(j["id_in_frame"]) and i["location"]==j["location"]:
                        #current_to_stay.append(i)
                        ids_to_stay_prev.append(id_prev)
                        ids_to_stay.append(idx)
                
            #we make sure what is in current of frame t and in previous in frame t+1 are the same objects identicallly 
            #ids_to_stay_prev.sort()# on trie parce que dans strackpool les detection sont récupérés dans l'ordre des matching avec les frames précédentes dans la fonction update de bytetrack.py
        
            dbn_infos[str(frame_id)]["current"]=np.array(dbn_infos[str(frame_id)]["current"])[ids_to_stay].tolist()#current_to_stay 
            dbn_infos[str(int(frame_id)+1)]["previous"]=np.array([dbn_infos[str(int(frame_id)+1)]["previous"][id_prev] for id_prev in ids_to_stay_prev]).tolist()#current_to_stay 
            
            for key in  dbn_infos[str(frame_id)]["observation"].keys(): 
                dbn_infos[str(frame_id)]["observation"][key]= np.array(dbn_infos[str(frame_id)]["observation"][key])[ids_to_stay].tolist()
            #if "observed" in list(dbn_infos[str(frame_id)].keys()):
            #    dbn_infos[str(frame_id)]["observation"]=np.array(dbn_infos[str(frame_id)]["observation"])[ids_to_stay].tolist()
            
            #on s'assure de transformer chaque ligne de la trice de transition en distribution de probabilités et se rassurer que current at t is previous at t+1
            dbn_infos[str(frame_id)]["matrice"] = np.array(dbn_infos[str(frame_id)]["matrice"])[:,ids_to_stay]
            dbn_infos[str(int(frame_id)+1)]["matrice"] =np.array(dbn_infos[str(int(frame_id)+1)]["matrice"] )
            dbn_infos[str(int(frame_id)+1)]["matrice"] = dbn_infos[str(int(frame_id)+1)]["matrice"][ids_to_stay_prev,:]
            dbn_infos[str(frame_id)]["matrice"] = 1- dbn_infos[str(frame_id)]["matrice"]
            
            
            for idx,_ in enumerate(dbn_infos[str(frame_id)]["matrice"]):
                if dbn_infos[str(frame_id)]["matrice"][idx].sum()!=0:
                    dbn_infos[str(frame_id)]["matrice"][idx]= dbn_infos[str(frame_id)]["matrice"][idx]/dbn_infos[str(frame_id)]["matrice"][idx].sum()
            
            dbn_infos[str(int(frame_id))]["matrice_inter"]=np.zeros((len(dbn_infos[str(int(frame_id))]["previous"]),len(dbn_infos[str(int(frame_id))]["current"])))
            for id_previous in range(len(dbn_infos[str(int(frame_id))]["previous"])):
                dbn_infos[str(int(frame_id))]["matrice_inter"][id_previous]= np.array([ iou(dbn_infos[str(int(frame_id))]["previous"][id_previous]["location"], dbn_infos[str(int(frame_id))]["current"][id_current]["location"]) for id_current in range(len(dbn_infos[str(int(frame_id))]["current"]))])
                dbn_infos[str(int(frame_id))]["matrice_inter"][id_previous] = dbn_infos[str(int(frame_id))]["matrice_inter"][id_previous]/dbn_infos[str(int(frame_id))]["matrice_inter"][id_previous].sum()
            
            dbn_infos[str(int(frame_id))]["matrice_difference"] = dbn_infos[str(int(frame_id))]["matrice_inter"] - dbn_infos[str(frame_id)]["matrice"]
            #if dbn_infos[str(int(frame_id))]["matrice_difference"].max()>0.3:
            #    print("grosse différence entre l'iou et le sort+appearance de bytetrack sur la frame",frame_id)


            dbn_infos[str(frame_id)]["matrice"] = np.array(dbn_infos[str(frame_id)]["matrice"]).tolist()
            dbn_infos[str(frame_id)]["matrice_inter"] = np.array(dbn_infos[str(frame_id)]["matrice_inter"]).tolist()
            for idx, track in enumerate(dbn_infos[str(frame_id)]["current"]):
                dbn_infos[str(frame_id)]["current"][idx]["track_id"]=dbn_infos[str(int(frame_id)+1)]["previous"][idx]["track_id"]

    with open(output_file, 'w') as outfile:
        json.dump(dbn_infos, outfile, default=lambda o: convert_to_json(o), indent=1)
        #print("the results txt files are #printed in", outfile)
        
        
        
#adding_atq()



