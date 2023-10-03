import pandas as pd 
import numpy as np
import os
import datetime as dt
import subprocess

# add the column with the nearest 10min starting 

file_feeder= "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/donnees_insentec_lot77_parc6.xlsx"
file_drinker = "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/eau_parc6.xlsx"
feeder_data = pd.read_excel(file_feeder)
#feeder_data.to_csv('xxx.csv')
drinker_data = pd.read_excel(file_drinker)
"""

drinker_data['rounded_Datetime'] = drinker_data['debut'].apply(lambda x: x - pd.to_timedelta(x.minute % 10, unit='m'))
drinker_data['rounded_Datetime']  = pd.to_datetime (drinker_data['rounded_Datetime']  ).dt.strftime('%Y-%m-%d %H:%M')
#print(drinker_data.head())

feeder_data['Date_fin']  = pd.to_datetime (feeder_data['Date_fin']  ).dt.strftime('%Y-%m-%d')
feeder_data['Tdebut']  = pd.to_datetime (feeder_data['Tdebut'],  format='%H:%M:%S'  ).dt.strftime('%H:%M')
feeder_data['Datetime'] =  pd.to_datetime(feeder_data['Date_fin'] + " "+ feeder_data['Tdebut'] )
feeder_data['rounded_Datetime'] = feeder_data['Datetime'].apply(lambda x: x - pd.to_timedelta(x.minute % 10, unit='m'))

#print(feeder_data.head())


# sum nbr of visit_per 10min_start of videos 
# select the 10min video the highest number of visit. The one selected:   2020-05-18 08:10:00   with   40 visites.  Done  ( 115612) pour le feeder, pour le drinker c'est
#for drinker it's  2020-05-12 09:10:00     51   (111320)
grouped_counts_feeder= feeder_data.groupby('rounded_Datetime').size().reset_index(name='count').sort_values(by="count", ascending=False)
grouped_counts_drinker = drinker_data.groupby('rounded_Datetime').size().reset_index(name='count').sort_values(by="count", ascending=False)
grouped_counts_drinker['rounded_Datetime'] = pd.to_datetime(grouped_counts_drinker['rounded_Datetime'])
grouped_counts_feeder['rounded_Datetime'] = pd.to_datetime(grouped_counts_feeder['rounded_Datetime'])
#specific_datetime = pd.to_datetime('2020-05-18 08:10:00', format='%Y-%m-%d %H:%M:%S')
#selected_rows = grouped_counts_drinker[(grouped_counts_drinker['rounded_Datetime'] == specific_datetime) ]
print(grouped_counts_drinker.head())
print(grouped_counts_feeder.head(4))

#download and past it in this folder 
exit(0)"""

################## tracking: find out the code doing detection on videos 


# Command to run
import sys
bytetrack_directory = "/home/sophie/uncertain-identity-aware-tracking/Bytetrack"
sys.path.append(bytetrack_directory)
video_path= "/home/sophie/uncertain-identity-aware-tracking/batch_identity_aware_tracking/115612-video.mp4"
video_debut=dt.datetime(2020, 5, 18, 8, 10,0) #to change 
video_fin = dt.datetime(2020, 5, 18, 8, 19,59) #to change 


#video_path="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314.mp4"
command = "python tools/demo_track_m.py video -f exps/example/mot/yolox_s_mix_det.py -c /home/sophie/uncertain-identity-aware-tracking/Bytetrack/models/yoloX_s_pig_trained_model_400_images.tar --path "+ video_path+"  --fuse --save_result --device gpu --fps 25 --conf 0.2 --track_thres 0.2  --match_thresh 1.0 --nms 0.45 --tsize 416"
print(command)
# Run the command
"""result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=bytetrack_directory)
# Check if the command was successful
if result.returncode == 0:
    print("Command executed successfully.")
    print("Output:")
    print(result.stdout)
else:
    print("Command failed with error:")
    print(result.stderr)
    
"""
############pass video start and end as parameter to the extract atq part of the code 
new_directory = "/home/sophie/uncertain-identity-aware-tracking/Bytetrack"
import sys
sys.path.append(new_directory)
from ATQ import adding_atq
from forwardBackward import process_forwad_backward

track_file=video_path.split('.mp4')[0]+'tracking_result.json'  ####
dbn_file=video_path.split('.mp4')[0]+'DBN_result.json'
observation_file=video_path.split('.mp4')[0]+"_result_with_observations_feeder.json"

#adding_atq(1, output_file=observation_file, feeder=True, video_debut=video_debut,video_fin= video_fin, track_file=track_file, dbn_file=dbn_file)

####ajouter tracking avec le HMM
HMM_result_file= video_path.split('.mp4')[0]+"_with_atq_tracking_with_HMM_result_feeder.json"
process_forwad_backward(observation_file,nbr_visit=1, json_save_path=HMM_result_file, video_path=video_path)



#rajouter les donner du drinker dans la fonction atq vu que j'utilise uniquement le feeder 

#faire un command line ou le mec guif la video et le feeder en command line et nous on gère