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
POWERFULNESS=1 #inverse of power of older observations
hungarian = True
confidence_threshold = 0.067  #### sophie mod 
confidence_on_hmm_choice=11  #1 equivaut à une normalisation basique #1.5

Home_folder=  "/home/sophie/uncertain-identity-aware-tracking/Bytetrack"
#en supposant que les observations sont independantes la normalisation à 1 des alpha et beta est acceptable 
#la solution qui suivra sera de choisir l'identité la plus acceptée au niveau de L et de l'affecté à la localisation identifié dans le tracking 


def process_forwad_backward(track_with_observation,nbr_visit="", json_save_path="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314_with_atq_tracking_with_HMM_result.json", video_path="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314.mp4"):
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
                #tmp_alpha = np.power(alpha[V[t-1]],POWERFULNESS)
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
                    #tmp_beta = np.power(beta[V[t+1]],POWERFULNESS)  #np.exp(alpha[V[t-1]])
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
            L["t="+str(t)] = L["t="+str(t)].tolist()
        ##print("final t", t)
            
        return L,beta,alpha
    

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

    identities_list= sorted(list(identities))

    V=[0]
    a={}
    b={}
    for identity in identities:
        b[str(identity)]={}
    initial_distribution = np.array([1/len(data["1"]["current"]) for i in data["1"]["current"] ])
    for frame_id, value in data.items():
        
        if  frame_id!="0" and int(frame_id)<max_frame: #int(frame)>100: #(frame!="0" and int(frame)%25==0) or
            a["t="+frame_id]=np.array(value["matrice"]) ######### without the hungarian choice hungarian_choice(np.array(value["matrice"]))

            """if 360<int(frame_id)<430:
                print(a["t="+frame_id].shape)
            """
            for identity in identities:
                b[str(identity)]["t="+frame_id]=np.array([1 for i in value["matrice"][0] ])

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


    for identity in identities_list [:15]:
        if True: # identity=='4808.0':
            L[identity], Beta[identity], Alpha[identity]= forward_backward_L(V=V,  a=a, b=b[str(identity)], initial_distribution=initial_distribution, T=V[-1] )#
            print(identity, "process finished")#"""
            



    ##################################Adding ATQ from the HMM #########################
    #Atq are added by considering the animal on which the confidence on an identity was greater than confidence_threshold
    
    """with open("data.json", 'w') as outfile:
        json.dump(data, outfile)
    
    with open("L.json", 'w') as outfile:
        json.dump(L, outfile)
        #exit(0)
        
    with open("data.json") as f:
            data = json.load(f)  
    
    with open("L.json") as f:
            L = json.load(f) 
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            return super(NumpyEncoder, self).default(obj)

    def get_proba_of_track_id(L,data):
        proba={}
        for identity in identities_list:
            proba[identity] ={i:[] for i in range(1,16,1)}
        proba["frame_id"]=[]
        for idx_t, t in enumerate( V[1:] ):
            proba['frame_id'].append(t)
            all_known_track = [i for i in range(1,16,1) ]
            for identity in identities_list:
                for idx, track in enumerate(data[str(t)]["current"]):
                    proba[identity][track["track_id"]].append(float(L[identity]["t="+str(t)][idx] ))
                    if track["track_id"] in all_known_track:
                        all_known_track.remove(track["track_id"])
                if all_known_track!=[]:
                    for remaining_track in all_known_track:
                        proba[identity][remaining_track].append(None)
        
        with open("proba.json", 'w') as outfile:
            json.dump(proba, outfile, cls=NumpyEncoder)
        print("proba dumped")
        return proba
    
    get_proba_of_track_id(L,data)
    
    
    for idx_t, t in enumerate( V[1:] ): 
        matrice=np.zeros((len(L[identities_list[0]]["t="+str(t)]),len(identities_list)))
        for idx,identity in enumerate(identities_list[:15]):
            matrice[:,idx]= L[identity]["t="+str(t)]

        #hungarian fin the correspondance with the minimal cost, since we want to maximize the  sum  of probabilities,  we will use -probability  
        """matrice_df={}
        for identity in identities:
                matrice_df[identity]= L[identity]["t="+str(t)]
        matrice_df=pd.DataFrame(matrice_df)"""

        ###faire le hungarian et on affiche slmt si le truc depasse le seuil
        if hungarian== True:
            #######Hungarian version whICH seems to be ok   
            try:
                #to avoid error in hungarian assignement we will replace nan by 0 those nan occures when the object is not present in one of the previous or next frame this is output in the forward backward process
                matrice =  np.nan_to_num(matrice, nan=0)
                row_ind, col_ind = linear_sum_assignment(-matrice) #since the function is looking for the assignement minimizing the sum, we put the opposite of the propabiliy in cells 
            except:
                print("xeption on this matrix")#,t,identities_list[3], L[identities_list[3]]["t="+str(t)], matrice)            
            for idx, row in enumerate(row_ind):
                data[str(t)]["current"][idx]["atq"] = identities_list[col_ind[idx]]
                
            for idx, track in enumerate(data[str(t)]["current"]):
                data[str(t)]["current"][idx]["atq"] = None
            for idx, row in enumerate(row_ind):
                if  L[identities_list[col_ind[idx]]]["t="+str(t)][idx]>confidence_threshold:
                        data[str(t)]["current"][idx]["atq"] = identities_list[col_ind[idx]]
                    

                
        else:
            ##########version with the maximum one feeting 
            for idx, track in enumerate(data[str(t)]["current"]):
                data[str(t)]["current"][idx]["atq"] = None
                identity_with_max_val= np.argmax(matrice[idx])
                #print(identity_with_max_val, identities_list[identity_with_max_val])
                if  L[identities_list[identity_with_max_val]]["t="+str(t)][idx]>confidence_threshold:
                    data[str(t)]["current"][idx]["atq"] = identities_list[identity_with_max_val]


          
    
    
    ######################################################
    #smoothing to make the tracker  take identities  from previous or future when it doesn't know what is the current identity 

    
    def get_track_from_id_and_time(track_id,t, gap=50, type="future"):
        """the function look in the past or future of the dataframe to see if there is an object with the same track_id having an atq near by
            la facon dont c'est implementé recommande un from previous avant from future 
        Args:
            track_id (_type_): _description_
            t (_type_): _description_
            gap (int, optional): _description_. Defaults to 50.
            type (str, optional): _description_. Defaults to "future".

        Returns:
            _type_: _description_
        """
        
        def check(t):
            for idx, track in enumerate(data[str(t)]["current"]):
                if track["track_id"]==track_id :
                    if track["atq"] is not None:
                        return track["atq"]
                    elif "atq_from_previous" in track.keys():
                        if track["atq_from_previous"]is not None:
                            return track["atq_from_previous"]
                        
                    elif "atq_from_future" in track.keys():
                        if track["atq_from_future"] is not None:
                            return track["atq_from_future"]
            return None
        t2 =t1=t
        while t1 in  V[t-gap:t] :#in data.keys():
            t= t1
            found_atq= check(t)
            if found_atq is not None :
                return check
            t= t2
            found_atq= check(t) 
                    #return None
            #print("ok**", t)
            t1=t1-1
            t2=t2+1
            #else:
            #    raise NameError("Type accepted are only future and past you provided "+type)
        return None

        
    def takens(current):
        takens=[]
        for idx, track in enumerate(current):
            takens.append(track["atq"])
            if "atq_from_previous" in track.keys():
                takens.append(track["atq_from_previous"])
            if "atq_from_future" in track.keys():
                takens.append(track["atq_from_future"])
                
        return takens 
           
    def smooting_from_past(data, gap=50):                
        """if atq is none get atq_previous:which is the atq of the animal having the same track_id in t-gap (it help relying on the tracker) 
        Args:
            data (_type_): _description_
            gap (int, optional): _description_. Defaults to 750.
        """
        for t in V[1:] : 
            takens_atq = takens(data[str(t)]["current"])
            
            
            for idx, track in enumerate(data[str(t)]["current"]):
                    if track["atq"] is None:
                        track_id=track["track_id"]
                        data[str(t)]["current"][idx]["atq_from_previous"]=  None
                        #track["atq_from_future"] =  None
                        if t>1+gap and V[-1]-t>gap: #"if not we can't do t-gap
                            #track_previous = get_track_from_id_and_time(track_id,t-1)
                            """if track_id==13:
                                atq_previous = get_track_from_id_and_time(track_id,t-1,gap=gap, type="past")
                                print(t, atq_previous)
                                exit(0)
                            """
                            atq_previous= None
                            atq_previous = get_track_from_id_and_time(track_id,t-1,gap=gap, type="past")
                            atq_previous =atq_previous if atq_previous not in takens_atq else None
                            if  atq_previous!=None:
                                data[str(t)]["current"][idx]["atq_from_previous"]= atq_previous #add check if atq not taken
                                takens_atq.append(atq_previous)
                            else:
                                data[str(t)]["current"][idx]["atq_from_previous"]=None
                                    
                            #except(e):
                        #    #print("an exception occur this is its description:",e)
                    #else:
                    #    data[str(t)]["current"][idx]["atq_from_previous"]= track["atq"]


    def smooting_from_future(data, gap=50):
        for t in reversed(V[1:]) : 
            takens_atq = takens(data[str(t)]["current"])
            
                
            for idx, track in enumerate(data[str(t)]["current"]):
                    if track["atq"] is None:
                        track_id=track["track_id"]
                        data[str(t)]["current"][idx]["atq_from_future"]= None
                        #track["atq_from_future"] =  None
                        if t>1+gap and V[-1]-t>gap: #"if not we can't do t-gap
                            atq_future= None
                            atq_future  = get_track_from_id_and_time(track_id,t+1,gap=gap, type="future")
                            if data[str(t)]["current"][idx]["atq_from_previous"]!=None:
                                takens_atq.append(data[str(t)]["current"][idx]["atq_from_previous"])
                            
                            atq_future = atq_future if atq_future not in takens_atq else None

                            if  atq_future is not None:
                                   data[str(t)]["current"][idx]["atq_from_future"]=  atq_future
                                   takens_atq.append(atq_future)
                            else:
                                data[str(t)]["current"][idx]["atq_from_future"]=None
                                    
                            #except(e):
                        #    #print("an exception occur this is its description:",e)
                    #else:
                    #    data[str(t)]["current"][idx]["atq_from_previous"]= track["atq"]


    #smooting_from_past(data,gap=10) # 1000)#
    # smooting_from_future(data,gap=10)#   4à secondes de gap

    


    ########################## Adding ATQ to the json file and the video 


    ##############writting on the video#########################

    
    ##print(tracks_with_atq)
    #video_path=Home_folder+"/videos/GR77_20200512_111314.mp4"
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_path=  video_path.split(".mp4")[0]+"_with_atq"+str(nbr_visit)+".mp4"
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
    track_file=video_path.split(".mp4")[0]+"tracking_result.json"  # Home_folder+"/videos/GR77_20200512_111314tracking_result.json"
    with open(track_file) as f:
            tracks = json.load(f) 
            

    ret_val, frame = cap.read()
    frame_id=1
    tracking_result={}
    while ret_val and frame_id<V[-1]: 
        dct={}
        ret_val, frame = cap.read()
        #add feeder and drinker center 
        frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
        frame = cv2.circle(frame, (90,102), radius, color, thickness)
        #addd
        if str(frame_id) in data.keys():
            if (frame_id!="0") :
                cv2.putText(frame, str(frame_id),(90+580, 20),0, 5e-3 * 200, (0,255,0),2)
                for idx,track in enumerate(data[str(frame_id)]["current"]):
                    track_id=track["track_id"] 
                    tlwh = track["location"]
                    #print(track)
                    atq= track["atq"] 
                    tid= str(track_id)   
                    if (atq is not None):
                        tid=tid+", atq:"+str(atq)+"("+str(round(L[atq]["t="+str(frame_id)][idx], 2))+")"
                    if  ("atq_from_previous" in track.keys() or "atq_from_future" in track.keys()  ) :
                        if track["atq_from_previous"] is not None:
                            #print("previous", track["atq_from_previous"] )
                            atq= track["atq_from_previous"]#+'fp'
                            tid=tid+", atq:"+str(atq)+"(fp)"#+str(round(L[track["atq_from_previous"]]["t="+str(frame_id)][idx], 2))+")"
                        """elif track["atq_from_future"] is not None:
                            #print("future",track["atq_from_future"] )
                            atq= track["atq_from_future"]#+'ff'#"""
                            #tid=tid+", atq:"+str(atq)+"("+str(round(L[track["atq_from_future"]]["t="+str(frame_id)][idx], 2))+")"""
                    #
                    
                    ##print(tid)
                    if atq is not None:
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
    print("ok done")
    #plutôt la surface d'intersection des rectangles plutôt que la distance eucledienne 




#process_forwad_backward()
#ajouter qui est au feeder sur la video
#rely on track  ligne 196 on des points intéréssant de roulement 