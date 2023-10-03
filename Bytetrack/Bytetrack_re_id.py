import pandas as pd 
import numpy as np
import json

track_with_observation_file= "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314DBN_result_with_observations_feeder.json"
re_id_track_result_file = "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314DBN_re_id.json"

re_id_video_file = "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314DBN_re_id.mp4"
video_path = "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314.mp4"


def produce_re_id_results(track_with_observation_file, re_id_track_result_file):
    with open(track_with_observation_file, 'r') as json_file:
        data = json.load(json_file)
        
    tracking_result={}
    observation_infos =[]
    matching={}
    corrected = {}
    #pour chaque ligne avec une observation garder le temps, atq, le track_id avec le max de chance d'être l'atq
    for frame_id, frame_infos in data.items():
        if frame_id!="0":
            dct={}
            if "observation" in frame_infos.keys():
                for atq in frame_infos["observation"].keys():
                    max_track_id =np.argmax(np.array(frame_infos["observation"][atq]))
                    track_id = frame_infos["current"][max_track_id]['track_id']
                    observation_infos.append((frame_id,atq,track_id))
                    if atq in matching.keys() : 
                        if matching[atq]!=track_id:
                            corrected[track_id] = matching[atq]
                            corrected[matching[atq]] = track_id
                            
                    matching[atq] = track_id
                    if matching[atq] in corrected.keys():
                                corrected[track_id] = corrected[matching[atq]]
                    if track_id in corrected.keys():
                        corrected[matching[atq]] = corrected[track_id]
        
            #we correct track_id by the observation
            for track in frame_infos["current"]:
                tlwh =track["location"]
                track_id = track["track_id"]
                if track["track_id"] in corrected.keys(): 
                    track_id  =  corrected[track["track_id"]]
                dct[track_id]=(int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3]) )
            
            tracking_result[frame_id]=dct

    with open(re_id_track_result_file, 'w') as outfile:
            json.dump(tracking_result, outfile)
    print("re-id is done")
    return tracking_result
    



def put_results_on_video(track, video_path, save_path, track_with_observation_file= track_with_observation_file):
    import cv2 
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 1, (int(width), int(height)))     
    # Center coordinates
    center_coordinates = (625, 70)
    # Radius of circle
    radius = 2
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    
    
    with open(track_with_observation_file, 'r') as json_file:
        data = json.load(json_file)
        
    ret_val, frame = cap.read()
    frame_id=1
    tracking_result={}
    
    while ret_val : 
        ret_val, frame = cap.read()
        #add feeder and drinker center 
        frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
        frame = cv2.circle(frame, (90,102), radius, color, thickness)
        #addd
        if frame_id in track.keys():
            
            if (frame_id!="0") :
                cv2.putText(frame, str(frame_id),(90+580, 20),0, 5e-3 * 200, (0,255,0),2)
                for track_id,tlwh in track[frame_id].items():
                    
                    tid= str(track_id)   
                    cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0])+int(tlwh[2]), int(tlwh[1])+int(tlwh[3])) ,(255,255,255), 2)
                    cv2.rectangle(frame, (580, 20), (90+580, 115+20) ,(255,255,255), 2)
                    cv2.putText(frame, str(tid),(int(tlwh[0]), int(tlwh[1])),0, 5e-3 * 200, (0,255,0),2)
                    
                vid_writer.write(frame)
        #print("\n", "\n")
        frame_id=frame_id+1
        #print("a frame done")
    vid_writer.release()
    print("video done")
    #plutôt la surface d'intersection des rectangles plutôt que la distance eucledienne 


import pandas as pd 
import numpy as np 
import json 
import plotly.express as px
import plotly.graph_objects as go

label_file= "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/labels_with_atq.json"
track_result = "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/GR77_20200512_111314tracking_result.json"
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
            track[frame][id]= tuple(detection)

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

"""label_file= "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/labels_without_atq.json"
label = read_data(label_file)
#put_results_on_video ( label , save_path="label_video.mp4" , video_path=video_path, track_with_observation_file=track_with_observation_file)
"""

#tracking_result =produce_re_id_results(track_with_observation_file, re_id_track_result_file)
#put_results_on_video ( tracking_result , save_path="videos/re_id_feeder_video.mp4" , video_path=video_path, track_with_observation_file=track_with_observation_file)