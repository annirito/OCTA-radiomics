# only basic columns from patient + First-order Features 

# The platform supports both the feature extraction in 2D and 3D and can be used to calculate single values per feature for a region of interest
#  (“segment-based”) or to generate feature maps (“voxel-based”).

import os
import sys
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from skimage.color import rgb2gray
import six

from radiomics import firstorder, glcm, glrlm, glszm, shape2D, ngtdm, gldm

import logging
# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
# ... or set level for specific class
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)

def sort_images(imgs):
    sorted_imgs = [None] * 490#  490 # * 700 for 6x6
    sorted_imgs_names = [None] *490 # * 490 for 3x3
    for img in imgs:
        # flow3x3
        # print('img:', img)
        if '_OD_' in img:
            number = (img[img.find('OD_')+3:img.rfind(".png")]) 
            if number in img:
                sorted_imgs[int(number)*2] = img
                sorted_imgs_names[int(number)*2] = 'R_{0}_3x3_'.format([int(number)])

        if '_OS_' in img:
            number = (img[img.find('OS_')+3:img.rfind(".png")]) 
            if number in img:
                sorted_imgs[int(number)*2+1] = img
                sorted_imgs_names[int(number)*2+1] = 'L_{0}_3x3_'.format([int(number)])
    return sorted_imgs, sorted_imgs_names



def extract_features(folder_path, folder, img, img_info):
    image = sitk.ReadImage(folder_path + '/' + folder + '/' + img, imageIO="PNGImageIO") # all images are png's

    # Convert image to gray-scale
    image_rgb = sitk.GetArrayFromImage(image)
    image_gray = rgb2gray(image_rgb)
    image = sitk.GetImageFromArray(image_gray)

    # plot image
    #slice = sitk.GetArrayFromImage(image)
    #plt.imshow(slice)
    #plt.show()

    mask = get_mask(root_dir, img_info, image)
    # mask = get_mask(root_dir, folder_img, img_info, image)

    settings = {'binWidth': 25,
                'interpolator': sitk.sitkBSpline,
                'resampledPixelSpacing': None}

    # FirstOrderFeatures
    firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)

    # firstOrderFeatures.enableFeatureByName('Mean', True)
    firstOrderFeatures.enableAllFeatures()

    # print('Will calculate the following first order features: ')
    # for f in firstOrderFeatures.enabledFeatures.keys():
    #    print('  ', f)
    #    print(getattr(firstOrderFeatures, 'get%sFeatureValue' % f).__doc__)

    # print('Calculating first order features...')
    results = firstOrderFeatures.execute()
    # print('done')

    first_order_features = dict()

    #print('Calculated first order features: ')
    for (key, val) in six.iteritems(results):
        first_order_features[img_info + key] = val
        #print('  ', key, ':', val)

    return first_order_features


#def get_mask(root_dir, folder_img, img_info, image):
def get_mask(folder_path, img_info, image):    
    mask_dir = root_dir + 'masks/' + img_info + '.nrrd'
    my_file = Path(mask_dir)
    if my_file.exists():
        mask = sitk.ReadImage(mask_dir, imageIO='NrrdImageIO') #Nrrd ("nearly raw raster data") is a 
        # library and file format for the representation and processing of n-dimensional raster data. 
        # It is intended to support scientific visualization and image processing applications.
    else:
        # Create mask ma
        # GetSize() return x, y, z size, array should be z, y, x. use [::-1] to reverse direction
        ma_arr = np.ones(image.GetSize()[::-1], dtype='int')
        # Get the SimpleITK image object from the array
        mask = sitk.GetImageFromArray(ma_arr)
        # Copy geometric information from the image (origin, spacing, direction)
        mask.CopyInformation(image)
        # Save mask
        sitk.WriteImage(mask, mask_dir, True)
        print('Mask created: ' + img_info)

    # plot mask
    #slice = sitk.GetArrayFromImage(ma)
    #plt.imshow(slice)
    #plt.show()

    return mask


def update_rows(row_R, row_L, first_order_features): 
    first_order_keys = list(first_order_features.keys())
    first_order_keys.sort()
    
    if img_info[0] == 'R':
        for key in first_order_keys:
            row_R.append(first_order_features[key])
            if int(folder) == 1:
                row_columns.append(key[2:])
        #print('FIRST ORDER: ' + str(len(first_order_keys)))
        # for key in shape_keys:
        #     row_R.append(shape_features[key])
        #     if int(folder) == 1:
        #         row_columns.append(key[2:])
        #print('2D: ' + str(len(shape_features)))
        
    elif img_info[0] == 'L':
        for key in first_order_keys:
            row_L.append(first_order_features[key])
            # if int(folder) == 1:
            #    row_columns.append(key[2:])
        
    else:
        print('ERROR: sorted_imgs_names')
        sys.exit()

    return row_R, row_L, row_columns


if __name__ == "__main__":
    #root_dir = '../data/'
    root_dir = '../dataset/'
    dataset_name = 'Maraton_OCTA_28062020'
    #folder_img = 'IMAGENES_ANONIMIZADAS'
    folder_path = '../folders/' 

    data = pd.read_excel(root_dir + dataset_name + '.xlsx', index_col=0,
                         usecols="A,B,E,G,J,N,"
                                 "CK,CL,CM,CN,CO,"
                                 "DC,DD,DE,DF,DG,"
                                 "DJ,DK,"
                                 "DL,DM,"
                                 "DR,DS,DV,DY,DZ,EA,EB,EC,ED,EE,EI,EM,EN,EO,EP,EQ,"
                                 "ET,EU,EX,FA,FB,FC,FD,FE,FF,FG,FK,FO,FP,FQ,FR,FS,"
                                 "GI,GJ,GK,GL,GM,GN,GO,GP,GQ,GR,GS,GT,GU,"
                                 "GV,GW,GX,GY,GZ,HA,HB,HC,HD,HE,HF,HG,HH")

    df = pd.DataFrame(data)  # Transform the dataset from the dataset to a dataset frame
    #df.columns = ['EDAD', 'OD', 'OI']
    df.columns = ['SEXO', 'DURACION_YEARS', 'EDAD', 'BMI', 'FUMADOR',
                  'PATOLOGIA_OCULAR_R', 'LASER_MACULAR_PREVIO_R', 'LASER_PRFC_PREVIO_R', 'Cx_PREVIAS_R', 'TTO_OCULAR_R',
                  'PATOLOGIA_OCULAR_L', 'LASER_MACULAR_PREVIO_L', 'LASER_PRFC_PREVIO_L', 'Cx_PREVIAS_L', 'TTO_OCULAR_L',
                  'CLASS_R', 'CLASS_L',
                  'EM_R', 'EM_L',
                  'CORRECTA_3X3_R', 'CALIDAD_3X3_R', 'VC_3X3_R', 'PC_3X3_R',  # 3X3 OCTA CORRECTA? (1=No, 2=Si), CALIDAD 3X3, VASO COMPLETA (mm-1), PERFUSION COMPLETA
                  'FAZ_3X3_R', 'A_3X3_R', 'P_3X3_R', 'C_3X3_R', # FAZ CORRECTA? (1=No, 2=Si, 3=no detecta), AREA (mm2), PERIMETRO, CIRCULARIDAD
                  'CORRECTA_6X6_R', 'CALIDAD_6X6_R', 'VC_6X6_R', 'PC_6X6_R', 
                  'FAZ_6X6_R', 'A_6X6_R', 'P_6X6_R', 'C_6X6_R',
                  'CORRECTA_3X3_L', 'CALIDAD_3X3_L', 'VC_3X3_L', 'PC_3X3_L',
                  'FAZ_3X3_L', 'A_3X3_L', 'P_3X3_L', 'C_3X3_L',
                  'CORRECTA_6X6_L', 'CALIDAD_6X6_L', 'VC_6X6_L', 'PC_6X6_L',
                  'FAZ_6X6_L', 'A_6X6_L', 'P_6X6_L', 'C_6X6_L',
                  'MAC_CORRECTO_R', 'MAC_CALIDAD_R', 'GC_R', 'V_R', 'PR_R', # OD MACULA CORRECTO? (1=No, 2=Si), MAC OD CALIDAD, GROSOR CENTRAL, VOLUMEN, PROMEDIO
                  'SE_R', 'NE_R', 'IE_R', 'TE_R', 'SI_R', 'NI_R', 'II_R', 'TI_R', # SUP EXT, NASAL EXT, INF EXT, TEMP EXT, SUP INT, NASAL INT, INF INT, TEMP INT
                  'MAC_CORRECTO_L', 'MAC_CALIDAD_L', 'GC_L', 'V_L', 'PR_L',
                  'SE_L', 'NE_L', 'IE_L', 'TE_L', 'SI_L', 'NI_L', 'II_L', 'TI_L']


    database = []

    folders = os.listdir(folder_path)
    #folders = os.listdir(root_dir + '/' + folder_img)
    folders.sort()

    row_columns = ['USER', 'SEXO', 'DURACION_YEARS', 'EDAD', 'BMI', 'FUMADOR', 'OJO', 'CLASE',
                   'PATOLOGIA_OCULAR', 'LASER_MACULAR_PREVIO', 'LASER_PRFC_PREVIO', 'Cx_PREVIAS', 'TTO_OCULAR', 'EM',
                   'CORRECTA_3X3', 'CALIDAD_3X3', 'VC_3X3', 'PC_3X3', 'FAZ_3X3', 'A_3X3', 'P_3X3', 'C_3X3',
                   'CORRECTA_6X6', 'CALIDAD_6X6', 'VC_6X6', 'PC_6X6', 'FAZ_6X6', 'A_6X6', 'P_6X6', 'C_6X6',
                   'MAC_CORRECTO', 'MAC_CALIDAD', 'GC', 'V', 'PR',
                   'SE', 'NE', 'IE', 'TE', 'SI', 'NI', 'II', 'TI']

    for folder in folders[:]: # [100:595] 
        if folder == '0062':
            print('FOLDER 0062')
            continue # skipping an iteration of a for loop with continue
        if folder == '0102':
            print('FOLDER 0102')
            continue
        if folder == '0137':
            print('FOLDER 0137')
            continue
        if folder == '0309':
            print('FOLDER 0309')
            continue
        if folder == '0592':
            print('FOLDER 0592')
            continue
        if folder == '0593':
            print('FOLDER 0593')
            continue
        if folder == '0597':
            print('FOLDER 0597')
            continue

        if folder[0] == '0': 
            print('folder: ', folder)
            user = int(folder)
            sexo = df.loc[int(folder)]["SEXO"]
            duracion_years = df.loc[int(folder)]["DURACION_YEARS"]
            edad = df.loc[int(folder)]["EDAD"]
            bmi = df.loc[int(folder)]["BMI"]
            fumador = df.loc[int(folder)]["FUMADOR"]

            ojo_R = 'R'
            if np.isnan(df.loc[int(folder)]["CLASS_R"]):
                y_R = ' '
            else:
                y_R = int(df.loc[int(folder)]["CLASS_R"])
            row_R = [user, sexo, duracion_years, edad, bmi, fumador, ojo_R, y_R,
                     df.loc[int(folder)]["PATOLOGIA_OCULAR_R"], df.loc[int(folder)]["LASER_MACULAR_PREVIO_R"],
                     df.loc[int(folder)]["LASER_PRFC_PREVIO_R"], df.loc[int(folder)]["Cx_PREVIAS_R"],
                     df.loc[int(folder)]["TTO_OCULAR_R"], df.loc[int(folder)]["EM_R"],
                     df.loc[int(folder)]["CORRECTA_3X3_R"], df.loc[int(folder)]["CALIDAD_3X3_R"],
                     df.loc[int(folder)]["VC_3X3_R"], df.loc[int(folder)]["PC_3X3_R"], df.loc[int(folder)]["FAZ_3X3_R"],
                     df.loc[int(folder)]["A_3X3_R"], df.loc[int(folder)]["P_3X3_R"], df.loc[int(folder)]["C_3X3_R"],
                     df.loc[int(folder)]["CORRECTA_6X6_R"], df.loc[int(folder)]["CALIDAD_6X6_R"],
                     df.loc[int(folder)]["VC_6X6_R"], df.loc[int(folder)]["PC_6X6_R"], df.loc[int(folder)]["FAZ_6X6_R"],
                     df.loc[int(folder)]["A_6X6_R"], df.loc[int(folder)]["P_6X6_R"], df.loc[int(folder)]["C_6X6_R"],
                     df.loc[int(folder)]["MAC_CORRECTO_R"], df.loc[int(folder)]["MAC_CALIDAD_R"],
                     df.loc[int(folder)]["GC_R"], df.loc[int(folder)]["V_R"], df.loc[int(folder)]["PR_R"],
                     df.loc[int(folder)]["SE_R"], df.loc[int(folder)]["NE_R"], df.loc[int(folder)]["IE_R"],
                     df.loc[int(folder)]["TE_R"], df.loc[int(folder)]["SI_R"], df.loc[int(folder)]["NI_R"],
                     df.loc[int(folder)]["II_R"], df.loc[int(folder)]["TI_R"]]


            ojo_L = 'L'
            if np.isnan(df.loc[int(folder)]["CLASS_L"]):
                y_L = ''
            else:
                y_L = int(df.loc[int(folder)]["CLASS_L"])
            row_L = [user, sexo, duracion_years, edad, bmi, fumador, ojo_L, y_L,
                     df.loc[int(folder)]["PATOLOGIA_OCULAR_L"], df.loc[int(folder)]["LASER_MACULAR_PREVIO_L"],
                     df.loc[int(folder)]["LASER_PRFC_PREVIO_L"], df.loc[int(folder)]["Cx_PREVIAS_L"],
                     df.loc[int(folder)]["TTO_OCULAR_L"], df.loc[int(folder)]["EM_L"],
                     df.loc[int(folder)]["CORRECTA_3X3_L"], df.loc[int(folder)]["CALIDAD_3X3_L"],
                     df.loc[int(folder)]["VC_3X3_L"], df.loc[int(folder)]["PC_3X3_L"], df.loc[int(folder)]["FAZ_3X3_L"],
                     df.loc[int(folder)]["A_3X3_L"], df.loc[int(folder)]["P_3X3_L"], df.loc[int(folder)]["C_3X3_L"],
                     df.loc[int(folder)]["CORRECTA_6X6_L"], df.loc[int(folder)]["CALIDAD_6X6_L"],
                     df.loc[int(folder)]["VC_6X6_L"], df.loc[int(folder)]["PC_6X6_L"], df.loc[int(folder)]["FAZ_6X6_L"],
                     df.loc[int(folder)]["A_6X6_L"], df.loc[int(folder)]["P_6X6_L"], df.loc[int(folder)]["C_6X6_L"],
                     df.loc[int(folder)]["MAC_CORRECTO_L"], df.loc[int(folder)]["MAC_CALIDAD_L"],
                     df.loc[int(folder)]["GC_L"], df.loc[int(folder)]["V_L"], df.loc[int(folder)]["PR_L"],
                     df.loc[int(folder)]["SE_L"], df.loc[int(folder)]["NE_L"], df.loc[int(folder)]["IE_L"],
                     df.loc[int(folder)]["TE_L"], df.loc[int(folder)]["SI_L"], df.loc[int(folder)]["NI_L"],
                     df.loc[int(folder)]["II_L"], df.loc[int(folder)]["TI_L"]]

            imgs = os.listdir(folder_path + '/' + folder)
            #imgs = os.listdir(root_dir + '/' + folder_img + '/' + folder)

            sorted_imgs, sorted_imgs_names = sort_images(imgs)

            for i, img in enumerate(sorted_imgs):
                #print(img)
                img_info = sorted_imgs_names[i]
                #print(img_info)

                if img_info != None:
                    # first_order_features, shape_features, glcm_features, glrlm_features, glszm_features, \
                    # ngtdm_features, gldm_features = extract_features(folder_path, folder, img, img_info)

                    first_order_features = extract_features(folder_path, folder, img, img_info)

                    # row_R, row_L, row_columns = update_rows(row_R, row_L, first_order_features, shape_features, glcm_features,
                    #                     glrlm_features, glszm_features, ngtdm_features, gldm_features)

                    row_R, row_L, row_columns = update_rows(row_R, row_L, first_order_features)


                if i == 1: # here: ??? if i == 1: but why ?
                    row_columns_ok = row_columns
            print(len(row_R))
            if len(row_R) == 4453: #25032: 3x3 # 35742: 6x6 +42
                database.append(row_R)
            else:
                print('NO OJO R ' + str(folder))
            print(len(row_L))
            if len(row_L) == 4453: # 23492 without NGTDM, GLDM, GLRLM # 33992 for 6x6 without NGTDM
                                    #  17892 without NGTDM, GLDM, GLRLM, GLSZM 
                                    # 14742 without NGTDM, GLDM, GLRLM, GLSZM, shape Features
                                    # 6342 only FirstOrderFeatures
                database.append(row_L)
            else:
                print('NO OJO L ' + str(folder))

    df = pd.DataFrame(database, columns=row_columns_ok) # having column names in csv
    #df = pd.DataFrame(database) # not having column names in csv
    df.to_csv(root_dir + 'raws3x3_FirstOrder.csv', index=False)