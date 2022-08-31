# class xcentre ycentre width height  <-- normalised values
############################## Computing accuracy using detect.py #######################################
import os
import shutil
import numpy as np
from collections import Counter
from sklearn import preprocessing
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as precision_recall
import matplotlib.pyplot as plt
from iptcinfo3 import IPTCInfo
import argparse

def create_dir(folder_path):
  if not os.path.isdir(folder_path):
    os.mkdir(folder_path)

def change_tags(image_path,image_name, save_tagged_path, keyword): #image path: str ; keyword: list
  # info = IPTCInfo('/content/group photo.jpg')
  try:
    info = IPTCInfo(os.path.join(image_path, image_name)+'.jpg')
    # add keyword       
    # info['keywords'] = ['Person','Vehicle']
    info['keywords'] = keyword
    # info.save()
    info.save_as(os.path.join(save_tagged_path, image_name)+'.JPG')
  except:
    info = IPTCInfo(os.path.join(image_path, image_name)+'.JPG')
    # add keyword       
    # info['keywords'] = ['Person','Vehicle']
    info['keywords'] = keyword
    # info.save()
    info.save_as(os.path.join(save_tagged_path, image_name)+'.JPG')

#creates dictionary of unique class name with their conf score in descending order
def unique_labels(list_pred):
  d={}
  for item in reversed(list_pred):
    class_num=item.split(' ')[0]
    conf=item.split(' ')[-1].split('\n')[0]
    # print(conf)
    # if (int(class_num) not in d) and (float(conf)>0.1): #conf threshold 
    if (int(class_num) not in d): #conf threshold 
      d.update({int(class_num):conf})
    else:
      continue
  return d


parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, help='Testing images path')
parser.add_argument('--pred_path', type=str, help= 'Path to predicted labels from detect.py')
parser.add_argument('--tagged_path', type=str, help='Path to tagged images directory')
FLAGS = parser.parse_args()

# original_images_path = '/home/ashimag/share_iiit_raw_autoseg_testing_24-8-2022/'
# images_path_bbox = '/home/ashimag/yolov5/runs/detect/yolo_test_24_08_correct_labels/'
# pred_path = '/home/ashimag/yolov5/runs/detect/yolo_test_24_08_correct_labels/labels/'

#path of original images with new tags######
# tagged_path="/home/ashimag/results/Tagged_images_test/"
create_dir(FLAGS.tagged_path)

classes = ['mani_cras-Manis crassicaudata', 'maca_munz-Macaca munzala', 'maca_radi-Macaca radiata', 'athe_macr', 'vulp_beng', 'lept_java-Leptoptilos javanicus',
 'trac_pile-Trachypithecus pileatus', 'hyst_brac-Hystrix brachyura', 'nilg_hylo-Nilgiritragus hylocrius', 'prio_vive-Prionailurus viverrinus',
  'neof_nebu-Neofelis nebulosa', 'melu_ursi', 'vehi_vehi', 'hyae_hyae-Hyaena hyaena', 'maca_mula-Macaca mulatta', 'fran_pond-Francolinus pondicerianus',
   'munt_munt-Muntiacus muntjak', 'feli_sylv-Felis sylvestris', 'maca_sile-Macaca silenus', 'vive_zibe-Viverra zibetha', 'rusa_unic-Rusa unicolor',
    'lepu_nigr-Lepus nigricollis', 'vive_indi-Viverricula indica', 'pavo_cris', 'anti_cerv', 'gall_lunu-Galloperdix lunulata', 'cato_temm-Catopuma temminckii',
     'sus__scro-Sus scrofa', 'cani_aure-Canis aureus', 'para_herm-Paradoxurus hermaphroditus', 'axis_axis', 'catt_kill', 'goat_sheep', 'vara_beng-Varanus bengalensis',
      'para-jerd-Paradoxurus jerdoni', 'mart_gwat-Martes gwatkinsii', 'homo_sapi', 'semn_john+Semnopithecus johnii', 'herp_edwa-Herpestes edwardsii', 'bos__fron',
       'herp_vitt-Herpestes vitticollis', 'arct_coll', 'dome_cats-Domestic cat', 'bos__indi', 'mell_cape-Mellivora capensis', 'ursu_thib-Ursus thibetanus',
        'semn_ente-Semnopithecus entellus', 'prio_rubi-Prionailurus rubiginosus', 'dome_dogs-Domestic dog', 'cani_lupu-Canis lupus', 'gall_sonn-Gallus sonneratii',
         'gaze_benn-Gazella bennettii', 'bose_trag-Boselaphus tragocamelus', 'budo_taxi-Budorcas taxicolor', 'bos__gaur', 'catt_catt-Cattle', 'blan_blan',
          'cuon_alpi-Cuon alpinus', 'capr_thar-Capricornis thar', 'equu_caba-Equus caballus', 'herp_fusc-Herpestes fuscus', 'trac_john-Trachypithecus johnii',
           'vara_salv-Varanus salvator', 'gall_gall-Gallus gallus', 'naem_gora-Naemorhedus goral', 'herp_urva-Herpestes urva', 'hyst_indi-Hystrix indica',
            'herp_smit-Herpestes smithii', 'bird_bird', 'tetr_quad-Tetracerus quadricornis', 'feli_chau-Felis chaus', 'maca_arct-Macaca arctoides',
             'lutr_pers-Lutrogale perspicillata', 'mosc_indi-Moschiola indica', 'pant_tigr', 'pant_pard-Panthera pardus', 'mart_flav-Martes flavigula',
              'pagu_larv-Paguma larvata-Masked Palm Civet', 'prio_beng-Prionailurus bengalensis', 'gall_spad-Galloperdix spadicea', 'elep_maxi-Elephas maximus',
               'axis_porc', 'anat_elli', 'bats_bats', 'call_pyge-Callosciurus pygerythrus', 'came_came-Camel', 'capr_hisp-Caprolagus hispidus', 'funa_palm-Funambulus palmarum',
                'hela_mala-Helarctos malayanus', 'lutr_lutr-Lutra lutra', 'maca_assa-Macaca assamensis', 'maca_leon-Macaca leonina', 'maca_maca-Macaque', 
                'melo_pers', 'pard_marm-Pardofelis marmorata', 'prio_pard-Prionodon pardicolor', 'tree_shre', 'vulp_vulp']   # class names

# classes = ['mani_cras-Manis crassicaudata', 'maca_munz-Macaca munzala', 'maca_radi-Macaca radiata', 'athe_macr', 'vulp_beng', 'lept_java-Leptoptilos javanicus',
#  'trac_pile-Trachypithecus pileatus', 'hyst_brac-Hystrix brachyura', 'nilg_hylo-Nilgiritragus hylocrius', 'prio_vive-Prionailurus viverrinus',
#   'neof_nebu-Neofelis nebulosa', 'melu_ursi', 'vehi_vehi', 'hyae_hyae-Hyaena hyaena', 'maca_mula-Macaca mulatta', 'fran_pond-Francolinus pondicerianus',
#    'munt_munt-Muntiacus muntjak', 'feli_sylv-Felis sylvestris', 'maca_sile-Macaca silenus', 'vive_zibe-Viverra zibetha', 'rusa_unic-Rusa unicolor',
#     'lepu_nigr-Lepus nigricollis', 'vive_indi-Viverricula indica', 'pavo_cris', 'anti_cerv', 'gall_lunu-Galloperdix lunulata', 'cato_temm-Catopuma temminckii',
#      'sus__scro-Sus scrofa', 'cani_aure-Canis aureus', 'para_herm-Paradoxurus hermaphroditus', 'axis_axis', 'catt_kill', 'goat_sheep', 'vara_beng-Varanus bengalensis',
#       'para-jerd-Paradoxurus jerdoni', 'mart_gwat-Martes gwatkinsii', 'homo_sapi', 'semn_john+Semnopithecus johnii', 'herp_edwa-Herpestes edwardsii', 'bos__fron',
#        'herp_vitt-Herpestes vitticollis', 'arct_coll', 'dome_cats-Domestic cat', 'bos__indi', 'mell_cape-Mellivora capensis', 'ursu_thib-Ursus thibetanus',
#         'semn_ente-Semnopithecus entellus', 'prio_rubi-Prionailurus rubiginosus', 'dome_dogs-Domestic dog', 'cani_lupu-Canis lupus', 'gall_sonn-Gallus sonneratii',
#          'gaze_benn-Gazella bennettii', 'bose_trag-Boselaphus tragocamelus', 'budo_taxi-Budorcas taxicolor', 'bos__gaur', 'catt_catt-Cattle', 'blan_blan',
#           'cuon_alpi-Cuon alpinus', 'capr_thar-Capricornis thar', 'equu_caba-Equus caballus', 'herp_fusc-Herpestes fuscus', 'trac_john-Trachypithecus johnii',
#            'vara_salv-Varanus salvator', 'gall_gall-Gallus gallus', 'naem_gora-Naemorhedus goral', 'herp_urva-Herpestes urva', 'hyst_indi-Hystrix indica',
#             'herp_smit-Herpestes smithii', 'bird_bird', 'tetr_quad-Tetracerus quadricornis', 'feli_chau-Felis chaus', 'maca_arct-Macaca arctoides',
#              'lutr_pers-Lutrogale perspicillata', 'mosc_indi-Moschiola indica', 'pant_tigr', 'pant_pard-Panthera pardus', 'mart_flav-Martes flavigula',
#               'pagu_larv-Paguma larvata-Masked Palm Civet', 'prio_beng-Prionailurus bengalensis', 'gall_spad-Galloperdix spadicea', 'elep_maxi-Elephas maximus',
#                'axis_porc']

le = preprocessing.LabelEncoder()
le.fit(classes)
word_to_int = le.transform(classes)
res = dict(zip(classes, word_to_int))

res['unid_unid'] = len(classes)

res_int_to_word = {}
keys = list(res.keys())
values = list(res.values())

for i in range(len(keys)):
    res_int_to_word[values[i]] = keys[i]

# sorted dictionary from int to word
sorted_res_int_to_word = OrderedDict(sorted(res_int_to_word.items()))


all_images = os.listdir(FLAGS.images_path)
all_pred_labels = os.listdir(FLAGS.pred_path)

# Check the original images provided and corresponding labels for those images.
print("Total images were {}, YOLO detected labels for {} images".format(len(all_images), len(all_pred_labels)))

############# segregate images in folders based on the detected label from txt files. 
# image_names = [file_name.split('.')[0] for file_name in all_images]
# first collect names of all images (remove the extension).
# create a dictionary mapping original image names with truncated names.
image_name_dict = {}
image_names = []
for file_name in all_images:
  image_name = file_name.split('.')[0]
  image_name_dict[image_name] = [file_name, '.'+file_name.split('.')[1]]      # save original file name and extension also
  image_names.append(image_name)



# iteratively look for predictions of these images and save preds in image_preds{} dictionary.
image_preds = {}
cnt = 0

for image in image_names:
  # import pdb; pdb.set_trace()
  try:
    text_file = image + '.txt'
    file = open(os.path.join(FLAGS.pred_path, text_file), 'r')
    all_preds = file.readlines()
    # pred = all_preds[-1]
    # pred_class = pred.split(' ')[0]
    pred_dict=unique_labels(all_preds)
    keyword=[]
    if len(all_preds)==0:  #for empty label.txt file, put the image in blank folder
      print('No label in txt')
      keyword=['blan_blan']
    elif len(pred_dict)==0: # if predictions less than confidence score put them in unidentified
      print('empty dict')
      # import pdb; pdb.set_trace()
      keyword.append('unid_unid')
    elif len(pred_dict.keys())<=3:    
      for label in pred_dict.keys():
        keyword.append(f'{sorted_res_int_to_word[label]}')
    elif len(pred_dict.keys())>3:
      count=1
      for label in pred_dict.keys():
        keyword.append(f'{sorted_res_int_to_word[label]}')
        count +=1
        if count>3:
          break
    else:

      print('#####################################')
      print('No condition met')
      print('#####################################')
    if len(keyword)==0:
      keyword.append('unid_unid')
    change_tags(FLAGS.images_path,image, FLAGS.tagged_path, keyword)
    print(f'Changed tags of {image} to {keyword}')

  except :
    print('Generated label not found for this image {}'.format(image))
    # assign blan_blan label for the images whose predictions were not generated by yolo. 
    cnt = cnt + 1
    keyword=['blan_blan']
    print(f'No of unidentified:{cnt}')
    change_tags(FLAGS.images_path, image, FLAGS.tagged_path,keyword)
    print(f'Changed tags of {image} to {keyword}')
    # may have to move these images in unid_unid folder later on.
    # image_preds[image] = res['unid_unid']

print("Total files without generated predictions are: ", cnt)
print("Created predictions dictionary")
