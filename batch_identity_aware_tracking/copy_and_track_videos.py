#conda activate Bytetrack
import dropbox
import os 
import subprocess
from pathlib import Path

def get_all_videos_in_subdirectories(root_path, video_extensions=['.mp4', '.avi', '.mkv']):
    video_files = []

    # Walk through all subdirectories
    for foldername, subfolders, filenames in os.walk(root_path):
        for filename in filenames:
            # Check if the file has a video extension
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(foldername, filename))

    return video_files

"""APP_KEY = "eklndt4zfwt60ng"
APP_SECRET = "6ceg2sqgg2s4l06"

oauth2_refresh_token="KXypH0WvX4YAAAAAAAAAAQJXPuuh10Cg4iqU8YmSMtrp8b8ff-1VwW_ySYjWeQD2"
dbx=dropbox.Dropbox(oauth2_refresh_token=oauth2_refresh_token,app_key=APP_KEY,app_secret=APP_SECRET)

# Accès aux données du serveur 252
tempresult = dbx.files_list_folder("/smarttracking/clients/deschambault/backup/serveur252")
"""
#Si ça peut être utilie, je joins une petite fonction pour lister le contenu d'un répertoire
#__________________________________________________________________________________
#
#   Function to list dropbox content at a given location. Use an existing dbx connection
#
#__________________________________________________________________________________

def db_content_listing(dbx,path,recursive=True,dirs_only=True,with_metadata=True,with_video_time=True ,extension_allowed=".mp4"):
    '''
    Function to list dropbox content at a given location. Use an existing dbx connection



    Parameters
    ----------
    dbx : dropbox object
      Established dropbox connecton

    path : str
        Path of folder in dropbox

    recursive : bool
        Tells if should walk all subdirectories

    dirs_only : bool
        Return directories list (if True), or files list (if False)

    with_metadata : bool
        If True, return file metadata

    with_video_time : bool
        If True, return video creation time extracted from file metadata

    Return : list of folders or folders and files. If with_metada or with_video_time return list of list
        containing requested information

        ex : with_metadata=False and with_video_time=False
            [file_path1,file_path2,...]

        ex : with_metadata=True and with_video_time=True
            [[file_path1,metadata1,creationtime1],[file_path2,metadata2,creationtime2],...]

    '''

    import dateutil
    import dropbox
    result=[]
    tempresult = dbx.files_list_folder(path, recursive=recursive)
    """for entry in tempresult.entries:
        if isinstance(entry, dropbox.files.FileMetadata)!=dirs_only:
            result.append(entry.path_lower)"""
    while tempresult.has_more:
        tempresult = dbx.files_list_folder_continue(tempresult.cursor)
        for entry in tempresult.entries:
            if isinstance(entry, dropbox.files.FileMetadata)!=dirs_only:
                if extension_allowed in entry.path_lower and".mp4" in entry.path_lower and "252/1" in entry.path_lower:
                  result.append(entry.path_lower)
                  
        
                

    if with_metadata or with_video_time:
        result_with_metadata=[]
        for file_link in result:
            file_metadata=dbx.files_get_metadata(file_link,include_media_info=True)
            temp_res=[file_link]
            if with_metadata:
                temp_res.append(file_metadata)
            if with_video_time:
                video_start_time = file_metadata.media_info.get_metadata().time_taken.replace(tzinfo=dateutil.tz.UTC)
                temp_res.append(video_start_time)
                break

    #         result_with_metadata.append(temp_res)
    #     return result_with_metadata

    #result =get_all_videos_in_subdirectories("/data/sophie/deschembault_videos")
    print(result)
    for dropbox_path in result:
      local_path = "/data/sophie/deschembault_videos"+dropbox_path.split("252/1")[1] #dropbox_path #
      video_name = dropbox_path.split("/")[-1]
      directory_path= local_path.split(video_name)[0]
      # Create the directory path, including any intermediate directories
      try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory path '{directory_path}' created successfully.")
        # Download the file from Dropbox to your local machine
        try:
            if not os.path.exists(local_path) and ".mp4" in local_path:
                with open(local_path, 'wb') as f:
                    metadata, res = dbx.files_download(dropbox_path)
                    f.write(res.content)
                print(f"Downloaded '{dropbox_path}' to '{local_path}'")
        except dropbox.exceptions.HttpError as err:
            print(f"Error downloading file: {err}")
      except OSError as e:
            print(f"Failed to create directory path '{directory_path}': {e}")
            
      ###we directely do the tracking 
      
    

    
      
      
    return result

#path= "/smarttracking/clients/deschambault/backup/serveur252/1/"
#db_content_listing(dbx,path,recursive=True,dirs_only=False,extension_allowed=".mp4",with_metadata=False,with_video_time=False)


def track_one(local_path):
     if os.path.exists(local_path) and ".mp4" in local_path : #we will only track videos of deschambaults 
        os.chdir("/home/sophie/uncertain-identity-aware-tracking/Bytetrack")
        print("starting track of "+ local_path)
        #subprocess.run("cd /home/sophie/uncertain-identity-aware-tracking/Bytetrack", shell=True)
        #subprocess.run("ls -l", shell =True)
        cmd = '''python3 tools/demo_track_m.py video -f exps/example/mot/yolox_s_mix_det.py -c'''
        cmd+="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/models/yoloX_s_pig_trained_model_400_images.tar --path "
        cmd+=local_path+ " --fuse --save_result --device cpu --fps 25 --conf 0.2 --track_thres 0.0  --match_thresh 1.0 --nms 0.45 --tsize 416 "
        print(cmd)
        subprocess.run(cmd,  shell =True)
        print("track ended")

def track_all():
    directory = "/data/sophie/deschembault_videos"
    for root, dirs, files in os.walk(directory):
        for file in files:
            if '.mp4' in file and 'result' not in file and "2020-05-12" in root and int(file.split("-")[0])>111314 and int(file.split("-")[0])<=111573  and not os.path.exists(root+"/"+file.split("mp4")[0]+"_tracking.mp4") : 
                local_path = root+"/"+file
                track_one(local_path)
                print(local_path)
                
                os.chdir("/home/sophie/uncertain-identity-aware-tracking/Bytetrack")
                print("starting track of "+ local_path)
                #subprocess.run("cd /home/sophie/uncertain-identity-aware-tracking/Bytetrack", shell=True)
                #subprocess.run("ls -l", shell =True)
                cmd = '''python3 tools/demo_track_m.py video -f exps/example/mot/yolox_s_mix_det.py -c'''
                cmd+="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/models/yoloX_s_pig_trained_model_400_images.tar --path "
                cmd+=local_path+ " --fuse --save_result --device gpu --fps 25 --conf 0.2 --track_thres 0.0  --match_thresh 1.0 --nms 0.45 --tsize 416 "
                print(cmd)
                subprocess.run(cmd,  shell =True)
                print("track ended")
        
                
        
   
        
track_all()