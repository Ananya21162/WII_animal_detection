import os
import shutil
import argparse  

parser = argparse.ArgumentParser()
parser.add_argument('--folders_path', type=str, help="path of folder containing subfolders of images.")
parser.add_argument('--merge_folder_path', type=str, help="path of folder to merge the images to.")
FLAGS = parser.parse_args()

# Function to create new folder if not exists
def make_new_folder(folder_name, parent_folder):
    # Path
    path = os.path.join(parent_folder, folder_name)
    # Create the folder 'new_folder' in parent_folder
    try: 
        # mode of the folder
        mode = 0o777
        # Create folder
        os.mkdir(path, mode) 
    except OSError as error: 
        print(error)


# list of folders to be merged
list_dir = []

def get_subdirs(dir):
    "Get a list of immediate subdirectories"
    return next(os.walk(dir))[1]

list_dir = get_subdirs(FLAGS.folders_path)
print(list_dir[0])
# enumerate on list_dir to get the content of all the folders ans store it in a dictionary
content_list = {}
for index, val in enumerate(list_dir):
    path = os.path.join(FLAGS.folders_path, val)
    content_list[list_dir[index]] = os.listdir(path)

# create merge_folder if not exists
previous_ = len(os.listdir(FLAGS.merge_folder_path))
print("no. of current train labels: ", previous_)

# loop through the list of folders
for sub_dir in content_list:
    # loop through the contents of the 
    # list of folders
    for contents in content_list[sub_dir]:
        # make the path of the content to move 
        path_to_content = sub_dir + "/" + contents  
        # make the path with the current folder
        dir_to_move = os.path.join(FLAGS.folders_path, path_to_content)
        # move the file
        shutil.copy(dir_to_move, FLAGS.merge_folder_path)

current_ = len(os.listdir(FLAGS.merge_folder_path))
print("Total no of files: ", current_)