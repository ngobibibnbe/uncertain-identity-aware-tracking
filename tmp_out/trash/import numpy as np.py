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

#en supposant que les observations sont independantes la normalisation à 1 des alpha et beta est acceptable 
#la solution qui suivra sera de choisir l'identité la plus acceptée au niveau de L et de l'affecté à la localisation identifié dans le tracking 

def softmax(x):
    #return x/sum(x)
    return np.exp(x)/sum(np.exp(x))
    #return np.power(x,2) / np.sum(np.power(x,2), axis=0)
    
  
def distance (boxA,boxB=[580, 0, 90+580, 115+20]):
    """
    This function process the distance between two boxes, by default it process the distance from the feeder it use the to corner left and bottom corner right
    in bytetrack our output are tlwh not tltr
    """
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
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou  #np.linalg.norm(np.array([float(track[0]), float(track[1])+float(track[3])/2])-np.array([600,17.5]))
    
def forward(V=np.array([0,1]), a={"t=1":np.array([[0.5,0.5]]) }, b={"t=0":np.array([0.5]),"t=1":np.array([0.5,0.5])}, initial_distribution=np.array([0.5]), T=1 ): #V=np.array([0,1]), a={"t=1":np.array([[0.5,0.5],[0.5,0.5]]) }, b={"t=0":np.array([0.5,0.5]),"t=1":np.array([0.5,0.5])}, initial_distribution=np.array([0.5,0.5])):
    alpha = {}
    alpha[V[1]] = initial_distribution * b["t="+str(V[1])]
    """for idx,value in enumerate(initial_distribution):
        initial_distribution[idx]=mpf(value)
      alpha[1]= matrix(b["t="+str(V[1])].shape[0], 1)
    for idx, value in enumerate(alpha[1]):
        alpha[1][idx]= mpf(initial_distribution[idx]) * mpf(b["t="+str(V[1])][idx])
    """
    for t in range(2, V.shape[0]):
        tmp_alpha = alpha[V[t - 1]]
        if "t="+str(V[t]) in list(b.keys()) and (b["t="+str(V[t])].max()==  b["t="+str(V[t])].min()) :#and (alpha[V[t-1]].max()== alpha[V[t-1]].min()) :
          #alpha[V[t]]=alpha[V[t-1]] #ca gère uniquement la première frame, il faudrait trouver un moyen de conserver les valeurs de alpha quand on a rien à la mangeoire
          #a_temp=np.ones(a["t="+str(V[t])].shape)
          alpha[V[t]]=np.zeros(a["t="+str(V[t])].shape[1])
          for j in range(a["t="+str(V[t])].shape[1]):
              alpha[V[t]][j] = tmp_alpha.dot((a["t="+str(V[t])][:, j])) * b["t="+str(V[t])][j] #scare a 
          final_alpha_t=alpha[V[t]]
          print(alpha[V[t]])
        else:
            tmp_b=b["t="+str(V[t])]
            if "t="+str(V[t]) in list(b.keys()) and  b["t="+str(V[t])].max()!=  b["t="+str(V[t])].min() :
                #si on a une observation on ramène le compteur de beta à 0 avec 1 partout à la qui suivais frame
                print("*****we rely on tracking")
                tmp_alpha= np.ones(alpha[V[t-1]].shape)

            alpha[V[t]]=np.zeros(a["t="+str(V[t])].shape[1])
            for j in range(a["t="+str(V[t])].shape[1]):
                alpha[V[t]][j] = tmp_alpha.dot(a["t="+str(V[t])][:, j]) *(b["t="+str(V[t])][j])
            final_alpha_t=alpha[V[t]]
            print(alpha[V[t]])

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
        if "t="+str(V[t]) in list(b.keys()) :#and  b["t="+str(V[t])].max()!=  b["t="+str(V[t])].min():
            """si on a une observation on ramène le compteur de beta à 0 avec 1 partout à la qui suivais frame"""
            #tmp_beta= np.ones(beta[V[t+1]].shape)
            tmp_beta=beta[V[t+1]]
            beta[V[t]]=np.zeros(a["t="+str(V[t+1])].shape[0])
            for j in range(a["t="+str(V[t+1])].shape[0]):
                beta[V[t]][j] = (tmp_beta * b["t="+str(V[t + 1])]).dot(a["t="+str(V[t+1])][j, :])

            beta[V[t]]=softmax(beta[V[t]])#sophie mod  beta[V[t]]/beta[V[t]].sum() #


          
    return beta
 
 
 
def forward_backward_L(V=np.array([0,1]), a={"t=1":np.array([[0.5,0.25]]) }, b={"t=0":np.array([0.5]),"t=1":np.array([0.5,0.5])}, initial_distribution=np.array([0.5]), T=1 ): #V=np.array([0,1]), a={"t=1":np.array([[0.5,0.5],[0.5,0.5]]) }, b={"t=0":np.array([0.5,0.5]),"t=1":np.array([0.5,0.5])}, initial_distribution=np.array([0.5,0.5])):):
  beta = backward(V,a,b,initial_distribution,T)
  alpha = forward(V,a,b,initial_distribution,T)
  L={}
  for t in V[1:]:
    if t==400:
      print(alpha[t],"***",beta[t])
    L["t="+str(t)]=alpha[t]*beta[t]
    if L["t="+str(t)].sum()!=0:
      L["t="+str(t)]=L["t="+str(t)]/L["t="+str(t)].sum()
    """if np.isnan(L["t="+str(t)]).any():
          o=beta[V[t]]
          p=alpha[V[t]]
          s=a["t="+str(V[t])]
          r=b["t="+str(V[t])]"""
  print("final t", t)
          
  return L,beta,alpha

import json
import numpy as np

with open('/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sophie/cdpq/ByteTrack/videos/GR77_20200512_111314DBN_resut_with_observations.json') as f:
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
    a["t="+frame_id]=np.array(value["matrice"])

    for identity in identities:
      b[str(identity)]["t="+frame_id]=np.array([1 for i in value["matrice"][1] ])

    for key in list(value["observation"]):
        b[str(key)]["t="+frame_id]=np.array(value["observation"][key] )
        if b[str(key)]["t="+frame_id].max()>0.1:
            print("observation at frame", frame_id, key)
        if frame_id=="549":
            print("************************************",b[str(key)]["t="+frame_id])
    #if 'observed' in list(value.keys()):
    #  b[str(value["observed"])]["t="+frame_id]=np.array(value["observation"])
    #quand l'animal n'est pas observé ca pourrait être un 1- la d istance à voir mais ca a des conséquence, l'ideal aurait été d'^tre sure quant à celui qui est à la mangeoire
    V.append(int(frame_id))


V=V[:3000]#-1]
V=np.array(V)
T=len(V)-2
L={}
Beta={}
Alpha={}


for identity in list(identities) [:15]:
    if identity=='4808.0':
        L[identity], Beta[identity], Alpha[identity]= forward_backward_L(V=V,  a=a, b=b[str(identity)], initial_distribution=initial_distribution, T=V[-1] )
        print(identity, "process finished")


####This part for matching  atq identities to objects ###########
import pandas as pd
plot_alpha={}
plot_beta={}
plot_L=pd.DataFrame(columns=[i for i in range(16)])
#print(plot)
#on va observer 4808