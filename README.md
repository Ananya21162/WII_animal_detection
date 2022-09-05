# WII_animal_detection
 Detecting animals in camera trap images.

### Steps To Run Detection On camera trap Images: 

1. To merge subfolders and create one separate folder, run the following command: <br />
```bash
python merge_folders.py --folders_path 'D:\images_directory\folders_with_subfolders'### path of folder to subfolders. 
                        --merge_folder_path 'D:\images_directory\merged_folder'     ### path of merged folder that contains all the images in one folder. 
```



2. To run detection on a new folder images_directory, run the following command: <br />
```bash 
python detect.py --source 'D:\images_directory\site0001'                             ### path to directory containing images (Note: Step 1 should be already completed.)
                 --weights runs/train/wii_28_072/weights/best.pt                     ### path to model weights.
                 --data data/wii_aite_2022_testing.yaml                              ### path to yaml file containing species names 
                 --img 640                                                           ### image size 
                 --save-txt                                                          ### save label txt files for every image.  
                 --save-conf                                                         ### saves confidences in label txt files.  
                 --name yolo_test_24_08_site0001                                     ### folder name created in ```runs/detect/``` with labels  
                 --conf-thres 0.001                                                  ### confidence threshold 0.001
                 --iou-thres 0.6                                                     ### iou_threshold 0.6
                 --empty_path D:\empty_files                                         ### path to save empty image files.
```
Arguments to change to run on different image directory includes : ```--source``` and ```--name```. 

To resume detection on images_directory and its final directory exists: <br /> 
```bash
python detect.py --source 'D:\images_directory\site0001'                             ### path to directory containing images (Note: Step 1 should be already completed.)
                 --weights runs/train/wii_28_072/weights/best.pt                     ### path to model weights.
                 --data data/wii_aite_2022_testing.yaml                              ### path to yaml file containing species names 
                 --img 640                                                           ### image size 
                 --save-txt                                                          ### save label txt files for every image.  
                 --save-conf                                                         ### saves confidences in label txt files.  
                 --name yolo_test_24_08_site0001                                     ### folder name created in ```runs/detect/``` with labels  
                 --conf-thres 0.001                                                  ### confidence threshold 0.001
                 --iou-thres 0.6                                                     ### iou_threshold 0.6
                 --resume                                                            ### resumes detection on the --source folder i.e. will generate predictions whose label files doesn't already exist. 
                 --exist-ok                                                          ### if output directory --name already exists and continue detection results to the same path
                 --empty_path D:\empty_files                                         ### path to save empty image files.
```
Arguments to additionally include to resume detection on different image directory includes : ```--resume``` and ```--exist-ok```. 




3. To create tags of the images with the generated labels, run tag_images.py. 
```bash
python tag_images.py --images_path 'D:\images_directory\site0001'                    ### path to images for testing
                     --pred_path 'runs/detect/yolo_test_24_08_site0001/labels/'      ### path where label .txt file is stored
                     --tagged_path 'D:\images_directory\site0001_tagged'             ### path where tagged images will be stored
```



