# only NGTDM_GLDM Features

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

    # NGTDM features
    ngtdmFeatures = ngtdm.RadiomicsNGTDM(image, mask, **settings)
    ngtdmFeatures.enableAllFeatures()

    results = ngtdmFeatures.execute()

    ngtdm_features = dict()

    for (key, val) in six.iteritems(results):
        ngtdm_features[img_info + key] = val

    # GLDM features
    gldmFeatures = gldm.RadiomicsGLDM(image, mask, **settings)
    gldmFeatures.enableAllFeatures()

    results = gldmFeatures.execute()
    # print('Will calculate the following gldm features: ')
    # for f in gldmFeatures.enabledFeatures.keys():
    #    print('  ', f)
    #    print(getattr(gldmFeatures, 'get%sFeatureValue' % f).__doc__)

    # print('Calculating shape features...')
    # results = gldmFeatures.execute()
    # print('done')

    gldm_features = dict()

    for (key, val) in six.iteritems(results):
        gldm_features[img_info + key] = val

    return gldm_features, ngtdm_features #, gldm_features # ngtdm_features, glrlm_features, , glszm_features shape_features,

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

def update_rows(row_R, row_L, ngtdm_features, gldm_features): 
    ngtdm_keys = list(ngtdm_features.keys())
    ngtdm_keys.sort()
    gldm_keys = list(gldm_features.keys())
    gldm_keys.sort()
    if img_info[0] == 'R':
        for key in ngtdm_keys:
            row_R.append(ngtdm_features[key])
            if int(folder) == 1:
                row_columns.append(key[2:])
        #print('ngtdm: ' + str(len(ngtdm_features)))
        for key in gldm_keys:
            row_R.append(gldm_features[key])
            if int(folder) == 1:
                row_columns.append(key[2:])
        #print('gldm: ' + str(len(gldm_features)))
    elif img_info[0] == 'L':
        for key in ngtdm_keys:
            row_L.append(ngtdm_features[key])
        #     # if int(folder) == 1:
        #     #    row_columns.append(key[2:])
        for key in gldm_keys:
            row_L.append(gldm_features[key])
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
    data = pd.read_excel(root_dir + dataset_name + '.xlsx', index_col=0, usecols="A,DJ:DK")

    df = pd.DataFrame(data)  # Transform the dataset from the dataset to a dataset frame
    #df.columns = ['EDAD', 'OD', 'OI']
    df.columns = ['CLASS_R', 'CLASS_L',]

    database = []

    folders = os.listdir(folder_path)
    #folders = os.listdir(root_dir + '/' + folder_img)
    folders.sort()

    row_columns = ['USER']

    for folder in folders[:]: # [100:595] #
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

            ojo_R = 'R'
            if np.isnan(df.loc[int(folder)]["CLASS_R"]):
                y_R = ' '
            else:
                y_R = int(df.loc[int(folder)]["CLASS_R"])
            row_R = [user]

            ojo_L = 'L'
            if np.isnan(df.loc[int(folder)]["CLASS_L"]):
                y_L = ''
            else:
                y_L = int(df.loc[int(folder)]["CLASS_L"])
            row_L = [user]

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

                    ngtdm_features, gldm_features = extract_features(folder_path, folder, img, img_info)

                    # row_R, row_L, row_columns = update_rows(row_R, row_L, first_order_features, shape_features, glcm_features,
                    #                     glrlm_features, glszm_features, ngtdm_features, gldm_features)

                    row_R, row_L, row_columns = update_rows(row_R, row_L, ngtdm_features, gldm_features)


                if i == 1: # here: ??? if i == 1: but why ? -> for column names 
                    row_columns_ok = row_columns

            print(len(row_R))
            if len(row_R) == 4656: #25032: 3x3 # 35742: 6x6
                database.append(row_R)
            else:
                print('NO OJO R ' + str(folder))
            print(len(row_L))
            if len(row_L) == 4656: # 23492 without NGTDM, GLDM, GLRLM # 33992 for 6x6 without NGTDM
                                    #  17892 without NGTDM, GLDM, GLRLM, GLSZM 
                                    # 14742 without NGTDM, GLDM, GLRLM, GLSZM, shape Features
                database.append(row_L)
            else:
                print('NO OJO L ' + str(folder))

    df = pd.DataFrame(database, columns=row_columns_ok)
    # df = pd.DataFrame(database)
    df.to_csv(root_dir + 'raws3x3_NGTDM_GLDM.csv', index=False)

