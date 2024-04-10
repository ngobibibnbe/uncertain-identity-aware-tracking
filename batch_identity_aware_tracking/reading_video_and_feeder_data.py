import dropbox
access_token="sl.BmQOSrXmwKduobc-cMkvOsF5ktPuKKBSdUyGyf1YU0hb5t1QY3c-8CBBuVnY5oJOBldrykKsnqJCLa3P_e9qFbeTfKqsN-29LPH8te7FrRKWgqjYPMDeGxYgGbRZuDJQQzusED6PBGNWDvPnOYIDW5k"

dbx = dropbox.Dropbox(access_token)

#dbx.users_get_current_account()
shared_folder_path = ''  # Update with the actual path
"""for entry in dbx.files_list_folder(shared_folder_path).entries:
    print(entry)"""


import requests

# Define the shared link URL
shared_link_url = 'https://www.dropbox.com/sh/k5igywxduqzk5b7/AADIZflzvkiQH9Pqh7oY409ba/serveur252/1/2020-03-04/68400?dl=0&preview=68400-video.mp4&subfolder_nav_tracking=1'  # Replace with your shared link URL
# Function to get file content from a shared link
def get_file_content_from_shared_link(link_url):
    try:
        # Parse the shared link to get the file ID
        shared_link_settings = dbx.sharing_get_shared_link_metadata(link_url)
        file_id = shared_link_settings.id

        # Download the file content
        _, response = dbx.files_download(file_id)

        # Assuming the content is text, you can access it using response.text
        content = response.text
        return content

    except dropbox.exceptions.ApiError as e:
        print(f"Error accessing shared link: {e}")

    return None

# Read the content from the shared link
file_content = get_file_content_from_shared_link(shared_link_url)

if file_content:
    print("File Content:")
    print(file_content)
else:
    print("Failed to access content from the shared link.")
    
    



###essayer rsync si syncdrpopbox ne marche pas

