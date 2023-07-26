import json
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


with open("/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sophie/cdpq/ByteTrack/videos/GR77_20200512_111314DBN_resut_good.json", 'r') as f:
  data = json.load(f)
  
edges=[]
nodes=[]
counter=0
nodes_evidences={}
for idx,key in enumerate(data.keys()):
  #if idx!=0:
  all_current_nodes=[]
  all_previous_nodes=[]
  higher_level_nodes=[]  
  for current_strack in data[key]['current']:
    current_node="O"+str(current_strack["frame_id"])+"-"+str(current_strack["id_in_frame"])
    all_current_nodes.append(current_node)
    nodes.append(current_node)
    nodes_evidences[current_node]={'previous':[],'higher':[]}

    for previous_strack in data[key]['previous']:
      previous_node="O"+str(previous_strack["frame_id"])+"-"+str(previous_strack["id_in_frame"])
      edges.append((previous_node,current_node) )
      nodes_evidences[current_node]['previous'].append(previous_node)
    for higher_level_node in higher_level_nodes:
      edges.append((higher_level_node,current_node))
      nodes_evidences[current_node]['higher'].append(higher_level_node)
    higher_level_nodes.append(current_node)

  for previous_strack in data[key]['previous']:
      previous_node="O"+str(previous_strack["frame_id"])+"-"+str(previous_strack["id_in_frame"])
      all_previous_nodes.append(previous_node)
  for current_strack in data[key]['current']:
    current_node="O"+str(current_strack["frame_id"])+"-"+str(current_strack["id_in_frame"])
    #print(len(all_previous_nodes),len(all_current_nodes),np.array(data[key]["matrice"]).shape)
    df=pd.DataFrame(data[key]["matrice"],columns=all_current_nodes, index=all_previous_nodes)
    nodes_evidences[current_node]["transition_matrice"]= 1-df
  #print(1-df)
  if counter==3:
    break
  counter+=1
print("********",edges)
#nodes=nodes[:20]


import numpy as np
import math
import copy
# Defining the CPDs:
cpds={}

cpds["O1-0"] = TabularCPD("O1-0",  16, [[0.997/15] for i in range(0,15)]+[[0.003]])
#nodes.remove("O1-0")
iter=0
for node in nodes: 
  if node!= "O1-0":
    evidences=nodes_evidences[node]["previous"] + nodes_evidences[node]["higher"]
    #np.zeros( (math.pow(16,int(len(evidences))),len(evidences)) )

    previous = nodes_evidences[node]["previous"]
    higher =nodes_evidences[node]["higher"]
    iterables = [[i for i in range(16)] for node in evidences]
    pd.MultiIndex.from_product(iterables, names=evidences)

    cols = pd.MultiIndex.from_product(iterables, names=evidences)
    df = pd.DataFrame([], columns=cols, index=[i for i in range(16)])

    for col in df.columns:
      original_col =copy.deepcopy(col)
      col=list(col)
      col_previous=np.array(list(col)[:len(previous)])
      col_higher=np.array(list(col)[-len(higher):])

      non_empty_col_previous =list(col)[:len(previous)]
      non_empty_col_higher =list(col)[-len(higher):] 
      #print("*******",col,non_empty_col_higher)
      if 15 in non_empty_col_higher:
        non_empty_col_higher=list(filter(lambda a: a != 15, non_empty_col_higher))
      if 15 in  non_empty_col_previous:
        non_empty_col_previous=list(filter(lambda a: a != 15, non_empty_col_previous))

      non_empty_col_previous = np.array(non_empty_col_previous)
      non_empty_col_higher =np.array(non_empty_col_higher)

      if (len(np.unique(non_empty_col_previous))!=len(non_empty_col_previous) or len(np.unique(non_empty_col_higher))!=len(non_empty_col_higher))  :
          #print("lklldksjlkdsjlk")
          #verifier si 2 elements de previous de col ont la même valeur si oui c'est impossible donc 0
          #de même verifier si deux éléments de higher ont la même valeur si oui c'est 0 
          df[original_col]="No"
          df.loc[15][original_col]=1
          continue
          #ok
      for index in df.index:
          if index in non_empty_col_higher:
            #si la valeur est la même qu'un autre alors ce n'est pas possible non plus
            df.loc[index][original_col]="No"
          else:
            if index!=15:
              interesting_previous=np.where(col_previous==index)[0]
              if len(interesting_previous)==0:
                if("O1-"  in node):
                  df.loc[index][original_col]=0.997/(15-len(non_empty_col_higher))
                else:
                  df.loc[index][original_col]=0
                
              else:
                interesting_previous = df.columns.names[interesting_previous[0]]
                #print(interesting_previous)
                df.loc[index][original_col]=nodes_evidences[node]["transition_matrice"][node].loc[interesting_previous] 
            else: 
              #here index==15
              df.loc[index][original_col]=0      
        
      

      if pd.to_numeric(df[original_col], errors='coerce').sum()<1:
        remaining =1-pd.to_numeric(df[original_col], errors='coerce').sum()
        df[original_col].loc[df[original_col]==0]=  remaining/len(df[original_col].loc[df[original_col]==0])
    #print(df)
    df = df.replace(['No'], 0)
    #print(pd.to_numeric(df[original_col], errors='coerce').sum())
      #else:
      #df.loc[index][original_col]=0.03 #1-acc model
      ##df.loc[index][original_col]=nodes_evidences[node]["transition_matrice"][node].loc[interesting_previous]    
          
        
        
    cpds[node] = TabularCPD(
        node,
        16,
        df.to_numpy(),
        evidence=evidences,
        evidence_card=[16 for node in nodes_evidences[node]['higher']+nodes_evidences[node]['previous']],
    )
    #print(df)
    iter =iter+1
    if iter==20:
      print(df)
      a = df.sum(axis=0)
      print(a.where(a!=1))
      break




#model.add_cpds(*cpds.values())
edges_i=edges#[:3]
model = BayesianNetwork(edges_i)

nodes_i =nodes[:20]
cpds_i= list(cpds.values())
cpds_i=cpds_i#[:3]
model.add_cpds(*cpds_i)

infer = VariableElimination(model)
posterior_p = infer.query(nodes_i)
print(posterior_p)