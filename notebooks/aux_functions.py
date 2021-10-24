# imports
import os
import pandas as pd
import csv
import numpy as np
import random
import itertools
import pickle
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc, confusion_matrix, classification_report
from matplotlib import pyplot as plt

def radiomics_mean(df, groups, slices):
    # getting Columnname of Radiomics 
    filter_col = [col for col in df if col.startswith('[0]')]
    #length_feature = len(filter_col)
    for i, col in enumerate(filter_col):
        filter_col[i] = col[4:]
    # looping through 
    df_mean = pd.DataFrame()  
    for col in filter_col:
        for group in range(groups): # 350/slices = groups
            df1 = [] # empty dataframe
            for slice in range(slices): # slices of groups
                slices_ = "[", str(slices*group+slice), "]_", col # slices = 5,10,14,25,35,50
                slices_ = ''.join(slices_) # for getting complete columnname
                cols = [col for col in df.columns if slices_ in col] # columnname: ['[0]_6x6_Autocorrelation'] 
                df_list = df[cols].values.tolist() # (index=False)
                df1.append(df_list)
            test = pd.DataFrame(df1)
            test = test.applymap(lambda x : x[0]) # getting rid of square brackets
            
            df_mean[col+'_mean_'+ str(group)] = test.mean(axis=0)     
    return df_mean


def radiomics_merge(directory, filepath, radiomic_mean, groups, slices):
    if radiomic_mean:
        df = pd.DataFrame()
        # looping through files in folder (in folder are all radiomic class features listed)
        for filename in os.listdir(directory):
            if filename.endswith(".csv"): 
                print(os.path.join(directory, filename))
                df_radiomics = pd.read_csv(os.path.join(directory, filename))
                df_mean = radiomics_mean(df_radiomics, groups, slices)
                df = pd.concat([df, df_mean], axis=1)

        # lastly add basic information of patient which are not radiomic features 
        df_basic = pd.read_csv(filepath + '.csv') # '.xlsx'
        df = pd.concat([df_basic, df], axis=1)
        # delete after: 
        print("type of df:", type(df))
        #print(df.head())

    else:
        df = pd.read_excel(directory + '.xlsx', index_col=0)  # Caution: here select file with all radiomic information for directory    
        #(index=False)
    return df     

def remove_GLCM_Sum_Average(df):
    print("def remove_GLCM_Sum_Average: ")
    for feature in df.columns:
        if 'SumAverage' in feature:
            #print("dropping Sum Average:", df[feature])
            df = df.drop([feature], axis=1)
    return df

def stringclean(df):
    print("stringclean ")
    df["DURACION_YEARS"] = df["DURACION_YEARS"].replace('', np.nan)
    df["DURACION_YEARS"] = df["DURACION_YEARS"].replace(' ', np.nan)
    df["DURACION_YEARS"] = df["DURACION_YEARS"].astype(float)
    
    df['PATOLOGIA_OCULAR'] = np.where(df.PATOLOGIA_OCULAR == '1*','1', df.PATOLOGIA_OCULAR)
    df['PATOLOGIA_OCULAR'] = np.where(df.PATOLOGIA_OCULAR == '3.4','3', df.PATOLOGIA_OCULAR) 

    df['Cx_PREVIAS'] = np.where(df.Cx_PREVIAS == '3, DR','3', df.Cx_PREVIAS) 
    df['Cx_PREVIAS'] = np.where(df.Cx_PREVIAS == '3,4','3', df.Cx_PREVIAS) 

    # make sure class (y) is int:
    df['CLASE'] = pd.to_numeric(df['CLASE'], errors='coerce')
    df.dropna(subset=['CLASE'], inplace=True) 

    return df


def remove_irrelevant_users(df):
    df["USER"] = df["USER"].replace('', np.nan)
    df["USER"] = df["USER"].replace(' ', np.nan)

    df.dropna(subset=['USER'], inplace=True) 

    return df    

def last_clean(df):
    print("last_clean ")
    # Iterate over the numeric attributes of the data
    for k in ['VC_3X3', 'PC_3X3', 'A_3X3', 'P_3X3', 'C_3X3', 'VC_6X6', 'PC_6X6', 'A_6X6', 'P_6X6', 'C_6X6', 'GC', 'V', 'PR', 'SE', 'NE', 'IE', 'TE', 'SI', 'NI', 'II', 'TI']:
        df[k] = df[k].replace('', np.nan)
        df[k] = df[k].replace(' ', np.nan)
        # Case 1: Some of the instances has this k attribute with 'na' value, but they are less than 50%
        if df[k].isna().any() and np.sum(df[k].isna()) / len(df[k]) < 0.5:
            # Change the missing values with the median of the rest of the values
            df[k] = df[k].fillna(df[k].median())
        # Case 2: More than 50% of the instances has this attribute with 'na' value
        if np.sum(df[k].isna()) / len(df[k]) >= 0.5:
            df.drop([k], axis=1, inplace=True)  # Drop column
            print('DROP COLUMN ' + k)   

    return df     

def impute_missing_numerical_onehot(df):
    print("impute_missing_numerical_onehot ")
    # Iterate over the numeric attributes of the data
    for k in ['DURACION_YEARS', 'EDAD', 'BMI']:

        # Case 1: Some of the instances has this k attribute with 'na' value, but they are less than 50%
        if df[k].isna().any() and np.sum(df[k].isna()) / len(df[k]) < 0.5:
            # Change the missing values with the median of the rest of the values
            df[k] = df[k].fillna(df[k].median())

        # Case 2: More than 50% of the instances has this attribute with 'na' value
        if np.sum(df[k].isna()) / len(df[k]) >= 0.5:
            df.drop([k], axis=1, inplace=True)  # Drop column
            print('DROP COLUMN ' + k)

    # Imput value 1 to missing values in 'FUMADOR' column and do one hot to this variable
    df['FUMADOR'].fillna(1)
    dfDummies = pd.get_dummies(df['FUMADOR'], prefix='FUMADOR')
    df = pd.concat([df, dfDummies], axis=1)
    df.drop(['FUMADOR'], axis=1, inplace=True)
    dfDummies = pd.get_dummies(df['SEXO'], prefix='SEXO')
    df = pd.concat([df, dfDummies], axis=1)
    df.drop(['SEXO'], axis=1, inplace=True)

    return df


def remove_irrelevant_cols(df): # hier nochmal ganz genau nachchecken welche Spalten gemeint sind
    print("remove_irrelevant_cols ")
    columns = [column for column in df.columns if ('3x3' in column)] 
    #  or '3x3' in column 
    for k in columns:
        if df[k].min() == df[k].max():
            df.drop(k, axis=1, inplace=True)
            
    return df


#TODO: 
def remove_instances(df, OCT_OCTA_exclusion_criteria, OCTA_filter, OCT_filter):
    total_index = []

    # Exclusion criteria (for software comercial OCTs y OCTAs)
    if OCT_OCTA_exclusion_criteria:
        # Patologia Ocular: remove >= 2
        df['PATOLOGIA_OCULAR'].astype(float)
        index_PO = df[df.PATOLOGIA_OCULAR.isin([2, 3, 4])].index
        #len(index_PO) = 68
        total_index.extend(index_PO)
        # Previous Laser Macular: remove 2
        index_LMP = df[df.LASER_MACULAR_PREVIO.isin([2])].index
        #len(index_LMP) = 21
        total_index.extend(index_LMP)
        # Previous Laser PRFC: remove 2
        index_LPRFCP = df[df.LASER_PRFC_PREVIO.isin([2])].index
        #len(index_LPRFCP) = 41
        total_index.extend(index_LPRFCP)
        # Previous Cx: remove >= 4
        index_Cx = df[df.Cx_PREVIAS.isin([3.4, 4, 5, 6])].index
        #len(index_Cx) = 9
        total_index.extend(index_Cx)
        # Tto ocular: remove 2
        index_TTO = df[df.TTO_OCULAR.isin([2])].index
        #len(index_TTO) = 24
        total_index.extend(index_TTO)
        # RESULTS: 1000 EYES aprox
        # Edema macular (EM): remove 2
        index_EM = df[df.EM.isin([2])].index
        #len(index_EM) = 13
        total_index.extend(index_EM)

    if OCTA_filter:
        # 3X3 OCTA correcta: remove 1 (artefactos)
        index_OCTA_3X3_correcta = df[df.CORRECTA_3X3.isin([1])].index
        #len(index_OCTA_3X3_correcta) = 67
        total_index.extend(index_OCTA_3X3_correcta)
        # 3X3 CALIDAD: remove < 7 (SSI)
        index_OCTA_3X3_calidad = df[df.CALIDAD_3X3.isin([1, 2, 3, 4, 5, 6])].index
        #len(index_OCTA_3X3_calidad) = 30
        total_index.extend(index_OCTA_3X3_calidad)
        # 3X3 FAZ CORRECTA: remove 1 and 3 (just keep 2)
        index_OCTA_3X3_faz = df[df.FAZ_3X3.isin([1, 3])].index
        #len(index_OCTA_3X3_faz) = 163
        total_index.extend(index_OCTA_3X3_faz)
        # 6X6 OCTA correcta: remove 1 (artefactos)
        index_OCTA_6X6_correcta = df[df.CORRECTA_6X6.isin([1])].index
        #len(index_OCTA_6X6_correcta) = 74
        total_index.extend(index_OCTA_6X6_correcta)
        # 6X6 calidad: remove < 7 (SSI)
        index_OCTA_6X6_calidad = df[df.CALIDAD_6X6.isin([1, 2, 3, 4, 5, 6])].index
        # len(index_OCTA_6X6_calidad) = 48
        total_index.extend(index_OCTA_6X6_calidad)
        # 6X6 FAZ correcta: remove 1 and 3 (just keep 2)
        index_OCTA_6X6_faz = df[df.FAZ_6X6.isin([1, 3])].index
        # len(index_OCTA_6X6_faz) = 226
        total_index.extend(index_OCTA_6X6_faz)

    if OCT_filter:
        # MAC correcta: remove 1
        index_MAC_correcta = df[df.MAC_CORRECTO.isin([1])].index
        # len(index_MAC_correcta) = 23
        total_index.extend(index_MAC_correcta)
        # MAC calidad: remove < 7 (SSI)
        index_MAC_calidad = df[df.MAC_CALIDAD.isin([1, 2, 3, 4, 5, 6])].index
        # len(index_MAC_calidad) = 38
        total_index.extend(index_MAC_calidad)

    total_index = list(set(total_index))
    print('total index length: ', len(total_index))
    # len(total_index) = 409

    df.drop(total_index, inplace=True)
    columns_filter = ['PATOLOGIA_OCULAR', 'LASER_MACULAR_PREVIO', 'LASER_PRFC_PREVIO', 'Cx_PREVIAS', 'TTO_OCULAR', 'EM',
                      'CORRECTA_3X3', 'CALIDAD_3X3', 'FAZ_3X3', 'CORRECTA_6X6', 'CALIDAD_6X6', 'FAZ_6X6',
                      'MAC_CORRECTO', 'MAC_CALIDAD']
    df.drop(columns_filter, axis=1, inplace=True)

    return df


def scale_numerical_attributes(df):
    # We apply the mean normalization (range [-1, 1])
    print("Def: scale_numerical_attributes")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for k in [col for col in df.columns if
              (('CLASE' not in col) and ('FUMADOR' not in col) and ('SEXO' not in col) and (col != 'USER') and (col != 'OJO'))]: # and ('SEXO' not in col)
        df[k] = scaler.fit_transform(df[[k]])

    print("Excel File als output")
    return df


def select_one_eye(df):
    random.seed(32)
    users = df['USER'].unique()
    for user in users:
        clases = df.loc[df['USER'].isin([user])]['CLASE'].tolist()
        if len(clases) > 1:
            index = df.loc[df['USER'].isin([user])].index.tolist()
            if len(set(clases)) > 1:
                eye_to_delete = clases.index(min(clases))
                df = df.drop([index[eye_to_delete]])
            else:
                df = df.drop([index[random.randint(0, 1)]])
    return df


def order_features_cv(n_splits, df, y, imgs, file_features, feature_type, labels, eyes): #, eyes):

    features = select_features(df, feature_type, imgs, labels)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=5)

    all_features_values = {}
    for i, (train_index, val_index) in enumerate(kf.split(df['USER'].unique())):
        features_values_dic = mutual_information(
            df.loc[df['USER'].isin(df['USER'].unique()[train_index])].copy(),
            y.loc[df['USER'].isin(df['USER'].unique()[train_index])].copy(), features)
        all_features_values[i] = features_values_dic

    features_csv_cols = ['split']
    features_csv_cols.extend(features)
    features_csv = [features_csv_cols]
    for split in all_features_values.keys():
        split_values = [split]
        for feature in features:
            if feature in all_features_values[split].keys():
                split_values.append(all_features_values[split][feature])
            else:
                split_values.append(0)
        features_csv.append(split_values)

    total_features = ['total']
    totalcols = np.array(features_csv)[1:, 1:].astype(np.float)
    sum = np.mean(totalcols, axis=0)
    total_features.extend(sum)
    features_csv.append(total_features)
    results_ = '../results_'
    with open(results_ + str(labels[0]) + '_' + str(labels[1]) + '_' + eyes + '/features_results_' + file_features + '.csv', "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(features_csv)

    features_total_values = []
    for pos, feature in enumerate(features):
        features_total_values.append((feature, sum[pos]))

    ordered_features_list = [feature for feature, val in sorted(features_total_values, key=lambda x: x[1], reverse=True)]

    return ordered_features_list


def select_features(df, feature_type, imgs, labels):

    columns = df.columns

    if feature_type in ['variables', 'all']:  # ['variables', 'radiomics-variables', 'software-variables', 'all']:
        features = ['EDAD', 'BMI', 'FUMADOR_1.0', 'FUMADOR_2.0', 'FUMADOR_3.0', 'SEXO_1', 'SEXO_2', 'VC_3X3', 'PC_3X3', 'A_3X3', 'P_3X3', 'C_3X3',
                             'VC_6X6', 'PC_6X6', 'A_6X6', 'P_6X6', 'C_6X6'] # , 'SEXO_1', 'SEXO_2']
        if labels == [1, 2]:
            features.append('DURACION_YEARS')
    else:
        features = []

    if feature_type in ['radiomics', 'all']: # , 'radiomics-variables', 'radiomics-software'
        features_ = [column for column in columns if '3x3' in column or '6x6' in column] # here: change string to 3x3 / ... 
        features_.sort()
        features.extend(features_)

    # Remove duplicate columns
    x = df[features].copy()
    features = x.T.drop_duplicates().T.columns

    return features


def mutual_information(df, y, features):
    x = df[features].copy()


    mi = list(enumerate(mutual_info_classif(x, y)))
    features_values = {}
    for value, feature in enumerate(x.columns):
        features_values[feature] = mi[value][1]

    return features_values


def dcv_search_params(kf5, kf4, model, df, y, labels, features):
    search_params = {'LogisticRegression': ({'C': np.logspace(-3, 3, 7), 'penalty': ['none','l2']}), # 'none', # np.logspace(-3, 3, 7) array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03])
                     #'LDA': ({'solver': ['svd', 'lsqr', 'eigen']}, LinearDiscriminantAnalysis()),
                     'LDA': ({'solver': ['svd', 'lsqr']}),
                     'SVClineal': ({'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.001, 0.0001]}),
                     'SVCrbf': ({'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.001, 0.0001]})}
    param_grid = search_params[model]
    keys, values = zip(*param_grid.items())
    params_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    auc_val_score_old = 0
    best_params = []
    max_iter = 1000
    for pram_combination in params_combinations:
        auc_val_score = dcv_evaluate_model(kf5, kf4, df, y, labels, model, features, pram_combination, max_iter)
        if auc_val_score > auc_val_score_old:
            auc_val_score_old = auc_val_score
            best_params = pram_combination

    return best_params, auc_val_score_old


def dcv_backward_elimination(kf5, kf4, model, df, y, labels, features, params, call_n, eyes, feature_type, features_name):
    old_features = features.copy()
    if model == 'LogisticRegression':
        model_sklearn = LogisticRegression(max_iter=10000, C=params['C'], penalty=params['penalty'])
    elif model == 'LDA':
        model_sklearn = LinearDiscriminantAnalysis(solver=params['solver'])
    elif model == 'SVClineal':
        model_sklearn = svm.SVC(kernel='linear', class_weight='balanced', max_iter=10000,
                                C=params['C'], gamma=params['gamma'], probability=True)
    elif model == 'SVCrbf':
        model_sklearn = svm.SVC(kernel='rbf', class_weight='balanced', max_iter=10000,
                                C=params['C'], gamma=params['gamma'], probability=True)

    max_iter = 10000
    old_auc_val_score = dcv_evaluate_model(kf5, kf4, df, y, labels, model, old_features, params, max_iter)
    num_features_list = [len(old_features)]
    features_list = [old_features]
    auc_val_list = [old_auc_val_score]

    print("len(old_features): ", len(old_features))
    while len(old_features) > 1:
        n_new_features = len(old_features)-1
        print("n_new_features: ", n_new_features) 
        sfs = SequentialFeatureSelector(model_sklearn, n_features_to_select=n_new_features, scoring='roc_auc') # Sequential Feature Selector adds (forward selection) or removes (backward selection) features to form a feature subset in a greedy fashion. At each stage, this estimator chooses the best feature to add or remove based on the cross-validation score of an estimator.
        # SequentialFeatureSelector direction default: forward
        sfs.fit(df[old_features], y)
        new_features = np.array(old_features)[sfs.get_support()].tolist() # get support: Get a mask, or integer index, of the features selected

        # Evaluate new features
        new_auc_val_score = dcv_evaluate_model(kf5, kf4, df, y, labels, model, new_features, params, max_iter)
        '''
        if new_auc_val_score >= old_auc_val_score:
            old_features = new_features
            old_auc_val_score = new_auc_val_score
        else:
            break
        '''
        num_features_list.append(n_new_features)
        features_list.append(new_features)
        auc_val_list.append(new_auc_val_score)
        if new_auc_val_score > old_auc_val_score:
            old_features = new_features
            old_auc_val_score = new_auc_val_score
        elif new_auc_val_score >= old_auc_val_score*0.98:
            old_features = new_features
        else:
            break

    plt.plot(num_features_list, auc_val_list)
    plt.xlabel('Num features')
    plt.ylabel('AUC validation')
    plt.title('AUC/features ' + model + ' ' + str(labels) + ' (' + str(call_n) + ')')
    #plt.show()
    results_ = '../results_'
    plt.savefig(results_ + str(labels[0]) + '_' + str(labels[1]) + '_' + eyes + '/features_wrapper_class_' + str(labels[0]) + '_' + str(labels[1]) + '_' + feature_type + '_' + features_name + '_' + model + '_' + str(call_n) + '.png')
    plt.close()

    df_features = pd.DataFrame(list(zip(num_features_list, auc_val_list, features_list)),
                               columns=['Num_features', 'AUC_val', 'features'])

    #with open("../results/features_wrapper_" + str(labels[0]) + '_' + str(labels[1]) + '_' + model + '_' + str(call_n) + ".csv", "w", newline="") as f:
    #    writer = csv.writer(f)
    #    writer.writerows(df_features)
    df_features.to_csv(results_ + str(labels[0]) + '_' + str(labels[1]) + '_' + eyes + '/features_wrapper_class_' + str(labels[0]) + '_' + str(labels[1]) + '_' + feature_type + '_' + features_name + '_' + model + '_' + str(call_n) + ".csv", index=False)

    best_auc_features = sorted(zip(auc_val_list, num_features_list, features_list), reverse=True)[:1]

    return best_auc_features[0][2]


def dcv_evaluate_model(kf5, kf4, df, y, labels, model, features, params, max_iter):
    roc_auc_val = []
    trainval_test = 0
    for trainval_index, test_index in kf5.split(df['USER'].unique()):
        #print('TRAINVAL - TEST SPLIT: ' + str(trainval_test), flush=True)
        train_val = 0
        for train_index, val_index in kf4.split(trainval_index):
            #print('TRAIN - VAL SPLIT: ' + str(train_val), flush=True)

            x = df[features].copy()

            # DEFINE TRAIN, VAL, TEST
            x_train = x.loc[df['USER'].isin(df['USER'].unique()[trainval_index[train_index]])].copy()
            y_train = y.loc[df['USER'].isin(df['USER'].unique()[trainval_index[train_index]])].copy()
            x_val = x.loc[df['USER'].isin(df['USER'].unique()[trainval_index[val_index]])].copy()
            y_val = y.loc[df['USER'].isin(df['USER'].unique()[trainval_index[val_index]])].copy()
            x_test = x.loc[df['USER'].isin(df['USER'].unique()[test_index])].copy()
            y_test = y.loc[df['USER'].isin(df['USER'].unique()[test_index])].copy()

            # CREATE MODEL
            if model == 'LogisticRegression':
                model_sklearn = LogisticRegression(max_iter=max_iter, C=params['C'], penalty=params['penalty'])
                classifier_probs = None
            elif model == 'LDA':
                model_sklearn = LinearDiscriminantAnalysis(solver=params['solver'])
                classifier_probs = None
            elif model == 'SVClineal':
                model_sklearn = svm.SVC(kernel='linear', class_weight='balanced', max_iter=max_iter,
                                        C=params['C'], gamma=params['gamma'])
                classifier_probs = svm.SVC(kernel='linear', class_weight='balanced', max_iter=max_iter,
                                           C=params['C'], gamma=params['gamma'], probability=True)
            elif model == 'SVCrbf':
                model_sklearn = svm.SVC(kernel='rbf', class_weight='balanced', max_iter=max_iter,
                                        C=params['C'], gamma=params['gamma'])
                classifier_probs = svm.SVC(kernel='rbf', class_weight='balanced', max_iter=max_iter,
                                           C=params['C'], gamma=params['gamma'], probability=True)
            else:
                print('MODEL ERROR')

            if model[:3] == 'SVC':
                y_prob = classifier_probs.fit(x_train, y_train).predict_proba(x_val)
            else:
                model_sklearn.fit(x_train, y_train)
                y_prob = model_sklearn.predict_proba(x_val)
            y_true = y_val

            if labels == [1, 2]:
                fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1], pos_label=2)
            else:
                fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)

            roc_auc_val.append(roc_auc)

            train_val += 1

        trainval_test += 1

    auc_val = mean(roc_auc_val)

    return auc_val


def evaluate_final_model(kf5, kf4, df, y, labels, model, feature_type, features, best_features, best_params, run_name,
                         save_cr, save_cm, save_rc, eyes):
    csvfile = [['trainval-test-split', 'train-val-split',
                'val_accuracy', 'val_accuracy_average_classes',
                'val_accuracy_class' + str(labels[0]), 'val_accuracy_class' + str(labels[1]),
                'val-precision_weighted', 'val-precision', 'val-specificity', 'val-recall', 'val-f1score', 'val-roc',
                'test_accuracy', 'test_accuracy_average_classes',
                'test_accuracy_class' + str(labels[0]), 'test_accuracy_class' + str(labels[1]),
                'test-precision_weighted', 'test-precision', 'test-specificity', 'test-recall', 'test-f1score', 'test-roc']]

    trainval_test = 0
    for trainval_index, test_index in kf5.split(df['USER'].unique()):
        #print('TRAINVAL - TEST SPLIT: ' + str(trainval_test), flush=True)
        train_val = 0
        for train_index, val_index in kf4.split(trainval_index):
            #print('TRAIN - VAL SPLIT: ' + str(train_val), flush=True)
            file_name = 'cv_' + str(trainval_test) + '_' + str(train_val)

            results, best_cols = create_evaluate_model(df, y, labels, model,
                                                       trainval_index, test_index, train_index, val_index,
                                                       trainval_test, train_val, run_name, file_name,
                                                       best_features, best_params,
                                                       save_cr, save_cm, save_rc, eyes)
            csvfile.append(results)

            train_val += 1

        trainval_test += 1

    total_metrics = ['total (mean)', run_name]
    npcsvfile = np.array(csvfile)[1:, 2:].astype(np.float)
    means = np.mean(npcsvfile, axis=0)
    final_means = [round(m, 5) for m in means]
    total_metrics.extend(final_means)
    csvfile.append(total_metrics)

    summary = [model, feature_type, features, len(best_features)]
    summary.extend(final_means)

    results_ = '../results_'
    with open(results_ + str(labels[0]) + '_' + str(labels[1]) + '_' + eyes + '/metric_results_' + run_name + '.csv', "w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csvfile)

    return summary


def create_evaluate_model(df, y, labels, model, trainval_index, test_index, train_index, val_index, i, j, run_name, file_name,
                          features, params, save_cr, save_cm, save_rc, eyes):

    best_cols_csv = [i, j]
    best_cols_csv.extend(features)

    x = df[features].copy()

    results = [i, j]

    # DEFINE TRAIN, VAL, TEST
    x_train = x.loc[df['USER'].isin(df['USER'].unique()[trainval_index[train_index]])].copy()
    y_train = y.loc[df['USER'].isin(df['USER'].unique()[trainval_index[train_index]])].copy()
    x_val = x.loc[df['USER'].isin(df['USER'].unique()[trainval_index[val_index]])].copy()
    y_val = y.loc[df['USER'].isin(df['USER'].unique()[trainval_index[val_index]])].copy()
    x_test = x.loc[df['USER'].isin(df['USER'].unique()[test_index])].copy()
    y_test = y.loc[df['USER'].isin(df['USER'].unique()[test_index])].copy()

    # CREATE MODEL
    if model == 'LogisticRegression':
        model_sklearn = LogisticRegression(max_iter=10000, C=params['C'], penalty=params['penalty'])
        classifier_probs = None
    elif model == 'LDA':
        model_sklearn = LinearDiscriminantAnalysis(solver=params['solver'])
        classifier_probs = None
    elif model == 'SVClineal':
        model_sklearn = svm.SVC(kernel='linear', class_weight='balanced', max_iter=10000,
                                C=params['C'], gamma=params['gamma'])
        classifier_probs = svm.SVC(kernel='linear', class_weight='balanced', max_iter=10000,
                                   C=params['C'], gamma=params['gamma'], probability=True)
    elif model == 'SVCrbf':
        model_sklearn = svm.SVC(kernel='rbf', class_weight='balanced', max_iter=10000,
                                C=params['C'], gamma=params['gamma'])
        classifier_probs = svm.SVC(kernel='rbf', class_weight='balanced', max_iter=10000,
                                   C=params['C'], gamma=params['gamma'], probability=True)
    else:
        print('MODEL ERROR')

    model_sklearn.fit(x_train, y_train)

    results_ = '../results_'
    path = results_ + str(labels[0]) + '_' + str(labels[1]) + '_' + eyes + '/' + run_name

    # VAL RESULTS
    measures_val, roc_auc_val = evaluate_val_test('val', model_sklearn, model, classifier_probs, labels, path, file_name,
                                                  x_train, y_train, x_val, y_val, save_cr, save_cm, save_rc)
    results.extend(measures_val)
    results.append(roc_auc_val)

    # TEST RESULTS
    measures_test, roc_auc_test = evaluate_val_test('test', model_sklearn, model, classifier_probs, labels, path, file_name,
                                                  x_train, y_train, x_test, y_test, save_cr, save_cm, save_rc)
    results.extend(measures_test)
    results.append(roc_auc_test)

    return results, best_cols_csv


def evaluate_val_test(val_train, model_sklearn, model, classifier_probs, labels, path, file_name, x_train, y_train, x, y,
                      save_cr, save_cm, save_rc):
    if val_train == 'val':
        extension = '_val'
    elif val_train == 'test':
        extension = '_test'


    if ('radiomics_retino' in path) and ('OCT' not in path):
        with open(path + '/model_sklearn_' + file_name + '.pickle', "wb") as ypred_model:
            pickle.dump(model_sklearn, ypred_model)

    y_pred = model_sklearn.predict(x)
    if model[:3] == 'SVC':
        model_probs = classifier_probs.fit(x_train, y_train)

        if ('radiomics_retino' in path) and ('OCT' not in path):
            with open(path + '/model_probs_' + file_name + '.pickle', "wb") as yprob_model:
                pickle.dump(model_probs, yprob_model)
        y_prob = model_probs.predict_proba(x)
    else:
        y_prob = model_sklearn.predict_proba(x)
    y_true = y

    if save_cr:
        cr = classification_report(y_true, y_pred)
        with open(path + '/Classification_Report_' + file_name + extension + '.txt', 'w') as f:
            f.write(cr)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if save_cm:
        plot_confusion_matrix(cm, labels=labels, path=path, file_name=file_name + extension)

    measures = return_measures(y_true.to_numpy(), y_pred, cm, path, file_name, extension)

    roc_auc = plot_roc_curve(y_true, y_prob, labels, path, file_name + extension, save_rc)

    return measures, roc_auc


def return_measures(y_true, y_pred, cm, path, file_name, extension):
    accuracy = accuracy_score(y_true, y_pred)
    accuracy = round(accuracy, 5)
    tn, fp, fn, tp = cm.ravel()
    cm_dic = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    # here:
    with open(path + '/cm_' + file_name + extension + '.pickle', "wb") as cm_dic_:
        pickle.dump(cm_dic, cm_dic_)
    #(tp + tn) / (tp + fp + fn + tn)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy_class1 = cm.diagonal()[0]
    accuracy_class2 = cm.diagonal()[1]
    accuracy_average = round((accuracy_class1+accuracy_class2)/2, 5)
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    precision_weighted = round(precision_weighted, 5)
    precision_binary = round(tp / (tp + fp), 5)
    specificity = round(tn / (tn + fp), 5)
    recall = round(tp / (tp + fn), 5)
    f1score = round(2*(recall * precision_binary)/(recall + precision_binary), 5)
    measures = [accuracy, accuracy_average, accuracy_class1, accuracy_class2, precision_weighted, precision_binary,
                specificity, recall, f1score]
    return measures


def plot_confusion_matrix(cm, labels, path, file_name, title='Confusion matrix', cmap=None):
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=cmap)
    plt.title(title)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    plt.xticks(fontsize=10)
    ax.set_yticklabels([''] + labels)
    plt.yticks(fontsize=10, rotation=90, va='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, str(z), ha='center', va='center', fontsize = 10)
    plt.savefig(path + '/confusion_matrix-' + file_name + '.png')
    plt.close()


def plot_roc_curve(y_true, y_prob, labels, path, file_name, save_rc):
    if labels == [1, 2]:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1], pos_label=2, drop_intermediate=False)
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1], pos_label=1, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    #print(fpr)
    #print(tpr)

    if save_rc:
        roc_dic = {'fpr': fpr, 'tpr': tpr}
        file_name_dic = 'roc_fpr_tpr_' + file_name
        with open(os.path.join(path, f"{file_name_dic}.pickle"), "wb") as roc_curve_dic:
            pickle.dump(roc_dic, roc_curve_dic)
        
        plt.clf()
        plt.plot(fpr, tpr, color='lightcoral', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc='best')
        plt.savefig(path + '/multiclass_roc-' + file_name + '.png', dpi=300)
        #plt.show()
        plt.close()
        

    return roc_auc


# ROC
# https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates