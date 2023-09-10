import json

def read_data(file):
    #here we will go through detections of deepsort 
    import json
    track={}
    with open(file) as f:
        json_file = json.load(f) 

    """for frame, detections in json_file.items():
        frame=int(frame)
        track[frame]={}
        for id, detection in detections.items():
            track[frame][id]={}
            track[frame][id]["rectangle"]= tuple(detection)
    """
    return json_file

track_1 = read_data("/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/data_1.json")
track_2 = read_data("/home/sophie/uncertain-identity-aware-tracking/Bytetrack/videos/data_2.json")

print(track_1==track_2)

for idx, value in track_1.items():
    if track_1[idx]!=track_2[idx]:
        t1= track_1[idx]
        t2=track_2[idx]
        print(idx, track_1[idx], "*****", track_2[idx])
        break
    #else:
    #    print("they are the same")