import numpy as np
from scipy.optimize import linear_sum_assignment
#! pip install mpmath
from mpmath import *
import pandas as pd
import numpy as np 
import datetime as dt
import json 
import copy
#!pip install opencv-python
import cv2
POWERFULNESS=2
Home_folder=  "/home/sophie/uncertain-identity-aware-tracking/Bytetrack"
#en supposant que les observations sont independantes la normalisation à 1 des alpha et beta est acceptable 
#la solution qui suivra sera de choisir l'identité la plus acceptée au niveau de L et de l'affecté à la localisation identifié dans le tracking 


def process_forwad_backward(track_with_observation,nbr_visit="", json_save_path="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314_with_atq_tracking_with_HMM_resut.json"):
    confidence_threshold = 0.3  #### sophie mod 
    confidence_on_hmm_choice=2#1.5
    """_summary_
    parameter: confidence_threshold

    Returns:
        write a video with ATQ and put the results in the file /home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314_with_atq_tracking_with_HMM_resut.json
    """
    import numpy as np 
    def softmax(x):
        #return x/sum(x)
        #return np.exp(x)/sum(np.exp(x))
        return np.power(x,confidence_on_hmm_choice) / np.sum(np.power(x,confidence_on_hmm_choice), axis=0)
        
    def hungarian_choice(matrice,  value_of_confidence_on_track=1):
        row_ind, col_ind = linear_sum_assignment(-matrice)
        for idx,row in enumerate(row_ind):
            matrice[row, col_ind[idx]] = value_of_confidence_on_track
        return matrice

    def forward(V=np.array([0,1]), a={"t=1":np.array([[0.5,0.5]]) }, b={"t=0":np.array([0.5]),"t=1":np.array([0.5,0.5])}, initial_distribution=np.array([0.5]), T=1 ): #V=np.array([0,1]), a={"t=1":np.array([[0.5,0.5],[0.5,0.5]]) }, b={"t=0":np.array([0.5,0.5]),"t=1":np.array([0.5,0.5])}, initial_distribution=np.array([0.5,0.5])):
        alpha = {}
        alpha[V[1]] = initial_distribution * b["t="+str(V[1])]
        for t in range(2, V.shape[0]):
            tmp_alpha = alpha[V[t - 1]]
            if "t="+str(V[t]) in list(b.keys()) and (b["t="+str(V[t])].max()==b["t="+str(V[t])].min()) and (alpha[V[t-1]].max()== alpha[V[t-1]].min()) :
                alpha[V[t]]=np.ones(a["t="+str(V[t])].shape[1]) #ca gère uniquement la première frame, il faudrait trouver un moyen de conserver les valeurs de alpha quand on a rien à la mangeoire
            else:
                tmp_b=b["t="+str(V[t])]
                if "t="+str(V[t]) in list(b.keys()) and  (b["t="+str(V[t])].max()==b["t="+str(V[t])].min()) and (alpha[V[t-1]].max()!= alpha[V[t-1]].min()):
                    tmp_alpha = np.power(alpha[V[t-1]],POWERFULNESS)  #np.exp(alpha[V[t-1]])
                
                if "t="+str(V[t]) in list(b.keys()) and  b["t="+str(V[t])].max()!=  b["t="+str(V[t])].min() :
                    #si on a une observation on ramène le compteur de beta à 0 avec 1 partout à la qui suivais frame
                    ##print("*****we rely on tracking only")
                    tmp_alpha= np.ones(alpha[V[t-1]].shape)

                alpha[V[t]]=np.zeros(a["t="+str(V[t])].shape[1])
                for j in range(a["t="+str(V[t])].shape[1]):
                    alpha[V[t]][j] = tmp_alpha.dot(a["t="+str(V[t])][:, j]) *(b["t="+str(V[t])][j])
                final_alpha_t=alpha[V[t]]
                
            ##print("*****",V[t], alpha[V[t]])
            if alpha[V[t]].sum()!=0:
                alpha[V[t]]=softmax(alpha[V[t]])
            
        return alpha
        
    def backward(V=np.array([0,1]), a={"t=1":np.array([[0.5,0.5]]) }, b={"t=0":np.array([0.5]),"t=1":np.array([0.5,0.5])},  initial_distribution=np.array([0.5]), T=1):
        beta = {}
        # setting beta(T) = 1
        beta[V[-1]] = np.ones((a["t="+str(V[-1])].shape[1]))

        # Loop in backward way from T-1 to
        # Due to python indexing the actual loop will be T-2 to 0
        for t in range(V.shape[0] - 2, -1, -1):
            #tmp_beta=beta[V[t+1]]
            #if "t="+str(V[t]) in list(b.keys()) and (b["t="+str(V[t])].max()==  b["t="+str(V[t])].min()) and  ( beta[V[t+1]].max()== beta[V[t+1]].min()):
            """ ca c'est lorsqu'on a pas d'observation, et que l'on vient de commencer avec beta, on préfère 
            laisser beta t1+1 à 1 et ne pas se laisser influencer par les probabilités de bytetrack
            ??? mais je ne suis pas sure que ca influence par ce que a est normaliser à 1 comme somme de cells par ligne??? à veirifier et tester """
            #  a_temp=np.ones(a["t="+str(V[t+1])].shape)
            #beta[V[t]]=beta[V[t+1]]
            #  beta[V[t]]=np.zeros(a["t="+str(V[t+1])].shape[0])
            #  for j in range(a["t="+str(V[t+1])].shape[0]):
            #    beta[V[t]][j] = (tmp_beta * b["t="+str(V[t + 1])]).dot(a_temp[j, :])
            #else:
            if "t="+str(V[t]) in list(b.keys()):
                tmp_beta=beta[V[t+1]]
                if  (b["t="+str(V[t+1])].max()==b["t="+str(V[t+1])].min()) and (beta[V[t+1]].max()== beta[V[t+1]].min()) :
                    beta[V[t]]=np.ones(a["t="+str(V[t+1])].shape[0]) #ca gère uniquement la première frame, il faudrait trouver un moyen de conserver les valeurs de alpha quand on a rien à la mangeoire
                else:
                    tmp_b=b["t="+str(V[t])]
                    if  (b["t="+str(V[t+1])].max()==b["t="+str(V[t+1])].min()) and (beta[V[t+1]].max()!= beta[V[t+1]].min()):
                        tmp_beta = np.power(beta[V[t+1]],POWERFULNESS)  #np.exp(alpha[V[t-1]])

                    if  b["t="+str(V[t+1])].max()!=  b["t="+str(V[t+1])].min() :
                        #si on a une observation on ramène le compteur de beta à 0 avec 1 partout à la qui suivais frame
                        ##print("*****we rely on tracking only", b["t="+str(V[t + 1])], tmp_beta * b["t="+str(V[t + 1])] )
                        tmp_beta= np.ones(beta[V[t+1]].shape)

                    beta[V[t]]=np.zeros(a["t="+str(V[t+1])].shape[0])
                    for j in range(a["t="+str(V[t+1])].shape[0]):
                        beta[V[t]][j] = (tmp_beta * b["t="+str(V[t + 1])]).dot(a["t="+str(V[t+1])][j, :])
                if beta[V[t]].sum()!=0:
                    beta[V[t]]=softmax(beta[V[t]])
                ##print("*****",V[t], beta[V[t]])
            
            """if "t="+str(V[t]) in list(b.keys()) :#and  b["t="+str(V[t])].max()!=  b["t="+str(V[t])].min():
                #si on a une observation on ramène le compteur de beta à 0 avec 1 partout à la qui suivais frame
                #tmp_beta= np.ones(beta[V[t+1]].shape)
                tmp_beta=beta[V[t+1]]
                beta[V[t]]=np.zeros(a["t="+str(V[t+1])].shape[0])
                for j in range(a["t="+str(V[t+1])].shape[0]):
                    beta[V[t]][j] = (tmp_beta * b["t="+str(V[t + 1])]).dot(a["t="+str(V[t+1])][j, :])

                beta[V[t]]=softmax(beta[V[t]])#sophie mod  beta[V[t]]/beta[V[t]].sum() #"""


            
        return beta
    
    
    
    def forward_backward_L(V=np.array([0,1]), a={"t=1":np.array([[0.5,0.25]]) }, b={"t=0":np.array([0.5]),"t=1":np.array([0.5,0.5])}, initial_distribution=np.array([0.5]), T=1 ): #V=np.array([0,1]), a={"t=1":np.array([[0.5,0.5],[0.5,0.5]]) }, b={"t=0":np.array([0.5,0.5]),"t=1":np.array([0.5,0.5])}, initial_distribution=np.array([0.5,0.5])):):
        beta = backward(V,a,b,initial_distribution,T)
        alpha = forward(V,a,b,initial_distribution,T)
        L={}
        for t in V[1:]:
            #if t==400:
            #  #print(alpha[t],"***",beta[t])
            L["t="+str(t)]=alpha[t]*beta[t] + alpha[t] + beta[t] #alpha[t]*beta[t]
            if L["t="+str(t)].sum()!=0:
                L["t="+str(t)]=L["t="+str(t)]/L["t="+str(t)].sum()

        ##print("final t", t)
            
        return L,beta,alpha

    import json
    import numpy as np

    with open(track_with_observation) as f:
        data = json.load(f)
    max_frame=max([int(i) for i in list(data.keys())])

    #### on crée la list des identités et ajoutons des noms au format identity'numéros' pour les inconus 
    identities=set()
    for frame, value in data.items():
        for key in value["observation"].keys():
            # if 'observed' in value.keys():
            identities.add(key)
    while len(identities)<15:
        identities.add("identities"+str(len(identities)))



    V=[0]
    a={}
    b={}
    for identity in identities:
        b[str(identity)]={}
    initial_distribution = np.array([1/len(data["0"]["current"]) for i in data["0"]["current"] ])
    for frame_id, value in data.items():
        if  frame_id!="0" and int(frame_id)<max_frame: #int(frame)>100: #(frame!="0" and int(frame)%25==0) or
            a["t="+frame_id]=np.array(value["matrice"]) ######### without the hungarian choice hungarian_choice(np.array(value["matrice"]))

            for identity in identities:
                b[str(identity)]["t="+frame_id]=np.array([1 for i in value["matrice"][1] ])

            for key in list(value["observation"]):
                b[str(key)]["t="+frame_id]=np.array(value["observation"][key] )
                #if b[str(key)]["t="+frame_id].max()>0.1:
                #    #print("observation at frame", frame_id, key)
                
            #if 'observed' in list(value.keys()):
            #  b[str(value["observed"])]["t="+frame_id]=np.array(value["observation"])
            #quand l'animal n'est pas observé ca pourrait être un 1- la d istance à voir mais ca a des conséquence, l'ideal aurait été d'^tre sure quant à celui qui est à la mangeoire
            V.append(int(frame_id))


    V=V[:-1]#3000]#-1]
    V=np.array(V)
    T=len(V)-2
    L={}
    Beta={}
    Alpha={}


    for identity in list(identities) [:15]:
        if True:#identity=='4809.0':
            L[identity], Beta[identity], Alpha[identity]= forward_backward_L(V=V,  a=a, b=b[str(identity)], initial_distribution=initial_distribution, T=V[-1] )
            #print(identity, "process finished")





    ##################################Adding ATQ from the HMM #########################
    #Atq are added by considering the animal on which the confidence on an identity was greater than confidence_threshold
    

    for idx_t, t in enumerate( V[1:] ): 
        matrice=np.zeros((len(L[list(identities)[0]]["t="+str(t)]),len(list(identities))))
        for idx,identity in enumerate(list(identities)[:15]):
            matrice[:,idx]= L[identity]["t="+str(t)]
        #hungarian fin the correspondance with the minimal cost, since we want to maximize the  sum  of probabilities,  we will use -probability  
        """matrice_df={}
        for identity in identities:
                matrice_df[identity]= L[identity]["t="+str(t)]
        matrice_df=pd.DataFrame(matrice_df)"""

        hungarian= False 
        if hungarian== True:
            #######Hungarian version whICH seems to be ok   
            try:
                row_ind, col_ind = linear_sum_assignment(-matrice) #since the function is looking for the assignement minimizing the sum, we put the opposite of the propabiliy in cells 
            except:
                print("xeption on this matrix")#,t,list(identities)[3], L[list(identities)[3]]["t="+str(t)], matrice)            
            for idx, row in enumerate(row_ind):
                data[str(t)]["current"][idx]["atq"] = list(identities)[col_ind[idx]]
                
        else:
            ##########version with the maximum one feeting 
            for idx, track in enumerate(data[str(t)]["current"]):
                track["atq"] = "None"
                identity_with_max_val= np.argmax(matrice[idx])
                #print(identity_with_max_val, list(identities)[identity_with_max_val])
                if  L[list(identities)[identity_with_max_val]]["t="+str(t)][idx]>confidence_threshold:
                    data[str(t)]["current"][idx]["atq"] = list(identities)[identity_with_max_val]


            

        
    ######################################################
    #smoothing to make the tracker  take identities  from previous or future when it doesn't know what is the current identity 

    
    def get_track_from_id_and_time(track_id,t):
        for idx, track in enumerate(data[str(t)]["current"]):
                if track["track_id"]==track_id:
                    return track

        
        
    def smooting_from_past(data, gap=750):
        """if atq is none get atq_previous:which is the atq of the animal having the same track_id in t-gap (it help relying on the tracker) 

        Args:
            data (_type_): _description_
            gap (int, optional): _description_. Defaults to 750.
        """
        for t in V[1:] : 
            for idx, track in enumerate(data[str(t)]["current"]):
                    if track["atq"]=="None":
                        track_id=track["track_id"]
                        track["atq_from_previous"]="None"
                        if t>1+gap: #"if not we can't do t-gap
                            #track_previous = get_track_from_id_and_time(track_id,t-1)
                            track_far_previous = get_track_from_id_and_time(track_id,t-gap)
                            if  track_far_previous!=None:
                                if  track_far_previous["atq_from_previous"]!="None":
                                    #if track_previous["atq_from_previous"]== track_far_previous["atq_from_previous"]:
                                    #print(t)
                                    #print(track)
                                    data[str(t)]["current"][idx]["atq_from_previous"]= track_far_previous["atq_from_previous"]
                                    ##print(track)
                            #except(e):
                        #    #print("an exception occur this is its description:",e)
                    else:
                        data[str(t)]["current"][idx]["atq_from_previous"]= track["atq"]


    def smooting_from_future(data, gap=750):
        for t in reversed(V[1:]) : 
            for idx, track in enumerate(data[str(t)]["current"]):
                    if track["atq"]=="None":
                        track_id=track["track_id"]
                        track["atq_from_future"]="None"
                        if V[-1]-t>gap:
                            track_future = get_track_from_id_and_time(track_id,t+1)
                            track_far_future = get_track_from_id_and_time(track_id,t+gap)

                            if track_future!=None and track_far_future!=None:

                                if track_future["atq_from_future"]!="None" and track_far_future["atq_from_future"]!="None":

                                    if track_future["atq_from_future"]== track_far_future["atq_from_future"]:
                                        #print(t)

                                        data[str(t)]["current"][idx]["atq_from_future"]= track_future["atq_from_future"]
                                        #print( data[str(t)]["current"][idx])
                                        ##print(track)
                            #except(e):
                        #    #print("an exception occur this is its description:",e)
                    #else:
                    #    data[str(t)]["current"][idx]["atq_from_future"]= track["atq"]

    smooting_from_future(data,gap=100) #=1000)#   4à secondes de gap
    smooting_from_past(data,gap=100) # 1000)#

    


    ########################## Adding ATQ to the json file and the video 


    ##############writting on the video#########################

    
    ##print(tracks_with_atq)
    video_path=Home_folder+"/videos/GR77_20200512_111314.mp4"
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_path= Home_folder+"/videos/GR77_20200512_111314_with_atq"+str(nbr_visit)+".mp4"
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))     




    # Center coordinates
    center_coordinates = (625, 70)
    
    # Radius of circle
    radius = 2
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    #frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
    
    
    #cv2.imwrite("image_.jpg", frame) 

    #print("********************** we start writting in the video")
    track_file=Home_folder+"/videos/GR77_20200512_111314tracking_resut.json"
    with open(track_file) as f:
            tracks = json.load(f) 
            

    ret_val, frame = cap.read()
    frame_id=1
    tracking_result={}
    while ret_val and frame_id!=V[-1]: 
        dct={}
        ret_val, frame = cap.read()
        frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
        if str(frame_id) in data.keys():
            if (frame_id!="0" ):
                cv2.putText(frame, str(frame_id),(90+580, 20),0, 5e-3 * 200, (0,255,0),2)
                for track in data[str(frame_id)]["current"]:
                    track_id=track["track_id"] 
                    tlwh = track["location"]
                    ##print(track)
                    atq= track["atq"] 
                    """if atq=="None" and "atq_from_previous" in track.keys() :
                        if track["atq_from_previous"]!="None":
                            atq= track["atq_from_previous"]#+'fp'
                        elif track["atq_from_future"]!="None":
                            atq= track["atq_from_future"]#+'ff'"""
                            
                    tid= str(track_id)+", atq:"+str(atq)
                    ##print(tid)
                    if atq!="None":
                        dct[atq]=(int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3]) )
                        ##print("we create the dct")
                    cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0])+int(tlwh[2]), int(tlwh[1])+int(tlwh[3])) ,(255,255,255), 2)
                    cv2.rectangle(frame, (580, 20), (90+580, 115+20) ,(255,255,255), 2)

                    cv2.putText(frame, str(tid),(int(tlwh[0]), int(tlwh[1])),0, 5e-3 * 200, (0,255,0),2)
                    
                tracking_result[frame_id]=dct
                vid_writer.write(frame)
        #print("\n", "\n")
        frame_id=frame_id+1
    with open(json_save_path, 'w') as outfile:
        json.dump(tracking_result, outfile)
    vid_writer.release()
    #plutôt la surface d'intersection des rectangles plutôt que la distance eucledienne 




#process_forwad_backward()