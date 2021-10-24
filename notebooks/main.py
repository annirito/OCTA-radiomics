# imports
import os
import warnings
from aux_functions import *


def loop_evaluation(test, features_vec, models, num_features, df, y, labels, save_cr, save_cm, save_rc, eyes):
    summary_results = [['model', 'features_type', 'features', 'num_features',
                        'val_accuracy', 'val_accuracy_average_classes',
                        'val_accuracy_class' + str(labels[0]), 'val_accuracy_class' + str(labels[1]),
                        'val-precision_weighted', 'val-precision', 'val-specificity', 'val-recall', 'val-f1score', 'val-roc',
                        'test_accuracy', 'test_accuracy_average_classes',
                        'test_accuracy_class' + str(labels[0]), 'test_accuracy_class' + str(labels[1]),
                        'test-precision_weighted', 'test-precision', 'test-specificity', 'test-recall', 'test-f1score', 'test-roc']]

    final_features_used = [['model', 'features_type', 'features', 'score',
                            'feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                            'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
                            'feature11', 'feature12', 'feature13', 'feature14', 'feature15',
                            'feature16', 'feature17', 'feature18', 'feature19', 'feature20',
                            'feature21', 'feature22', 'feature23', 'feature24', 'feature25',
                            'feature26', 'feature27', 'feature28', 'feature29', 'feature30',
                            'feature31', 'feature32', 'feature33', 'feature34', 'feature35',
                            'feature36', 'feature37', 'feature38', 'feature39', 'feature40',
                            'feature41', 'feature42', 'feature43', 'feature44', 'feature45',
                            'feature46', 'feature47', 'feature48', 'feature49', 'feature50',
                            'feature51', 'feature52', 'feature53', 'feature54', 'feature55',
                            'feature56', 'feature57', 'feature58', 'feature59', 'feature60'
                            ]]

    # Double CV definition
    kf5 = KFold(n_splits=5, shuffle=True, random_state=1)
    kf4 = KFold(n_splits=4, shuffle=True, random_state=3)

    # Input type analysis
    for feature_type in features_vec.keys():
        print('FEATURE TYPE: ' + feature_type, flush=True)

        # Input images
        for features in features_vec[feature_type]:
            features_name = ''
            for feature in features:
                features_name = features_name + feature
            print('INPUT FEATURES: ' + features_name, flush=True)

            n_splits = 10
            file_features = test + '_' + feature_type + '_' + features_name
            # Simple cross validation with n_splits splits
            features_ordered_list = order_features_cv(n_splits, df, y, features, file_features, feature_type, labels, eyes)

            # Model
            for model in models:
                print('MODEL: ' + model, flush=True)
                run_name = test + '_' + model + '_' + feature_type + '_' + features_name

                # Create folder
                path = results_ + str(labels[0]) + '_' + str(labels[1]) + '_' + eyes + '/' + run_name
                if not os.path.exists(path):
                    os.makedirs(path)

                if len(features_ordered_list) >= num_features:
                    # TODO: explain when this step is performed in the paper
                    initial_features = features_ordered_list[:num_features].copy()
                else:
                    initial_features = features_ordered_list.copy()

                initial_params, initial_auc_val_score = dcv_search_params(kf5, kf4, model, df, y, labels, initial_features)

                #print('START GRIT SEARCH - FEATURES WRAPPER LOOP')
                iteration = 0
                new_features = []
                new_params = None
                call_n = 0
                while not ((len(initial_features) == len(new_features)) and (initial_params == new_params)):
                    if len(initial_features) > 1:
                        new_features = dcv_backward_elimination(kf5, kf4, model, df, y, labels, initial_features, initial_params, call_n, eyes, feature_type, features_name)
                        call_n += 1
                    else:
                        new_features = initial_features
                    new_params, new_auc_val_score = dcv_search_params(kf5, kf4, model, df, y, labels, new_features)
                    iteration += 1
                    initial_features = new_features
                    initial_params = new_params

                #print('GRIT SEARCH - FEATURES WRAPPER LOOP FINISHED')
                #print('Iterations needed: ' + str(iteration))

                best_features = new_features
                best_params = new_params
                best_score = new_auc_val_score

                features_file = [model, feature_type, features, best_score]
                features_file.extend(best_features)
                features_file.extend([''] * 30)#60)
                final_features_used.append(features_file[:34])#64])

                summary = evaluate_final_model(kf5, kf4, df, y, labels, model, feature_type, features,
                                               best_features, best_params, run_name, save_cr, save_cm, save_rc, eyes)

                summary_results.append(summary)

    models_ = ''
    for model in models:
        models_ += model
    with open(results_ + str(labels[0]) + '_' + str(labels[1]) + '_' + eyes + '/summary_results_' + models_ + '_' + test + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(summary_results)

    with open(results_ + str(labels[0]) + '_' + str(labels[1]) + '_' + eyes + '/summary_features_' + models_ + '_' + test + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(final_features_used)


if __name__ == "__main__":
    # Define parameters
    test = 'class_0_1'
    rootdir = '../dataset/'
    results_ = '../results_'

    filepath = '../dataset/'

    #### directory if means will be calculated by aux_functions
    #directory = '../dataset/flow3x3'

    #### directory if means already calculated beforehand 
    directory = '../dataset/flow3x3'

    warnings.filterwarnings("ignore")

    features_vec = {'radiomics': [['retino']], 'all': [['variables', 'retino']]}

    models = ['LogisticRegression', 'LDA', 'SVClineal', 'SVCrbf']
    #models = ['LogisticRegression']

    num_features = 30   #20

    classes = '0-1'
    #classes = '1-2'

    # Save Confusion Matrix
    save_cm = True
    # Save Clasification Report
    save_cr = True
    # Save ROC Curve
    save_rc = True

    OCT_OCTA_exclusion_criteria = True
    OCTA_filter = True
    OCT_filter = False

    just_one_eye = False

    radiomic_mean = False
    # flow 6x6 = 350
    # groups = 5 # 70, 50, 35, 25, 14, 10, 7, 5
    # slices = 70 # 5, 7, 10, 14, 25, 35, 50, 70

    # flow 3x3 = 245
    groups = 1 # 49, 35, 7, 5
    slices = 245 # 5, 7, 35, 49


    #df = pd.read_csv(filepath + '.csv')
    #df = pd.read_excel(filepath + '.xlsx')

    # read df and caclulating mean first 
    df = radiomics_merge(directory, filepath, radiomic_mean, groups, slices)


    # Remove GLCM Sum Average
    df = remove_GLCM_Sum_Average(df)
    # Removing empty strings of "DuracionYears"
    df = stringclean(df)
    #
    df = last_clean(df)
    # Impute missing values
    df = impute_missing_numerical_onehot(df)
    # Remove irrelevant columns
    df = remove_irrelevant_cols(df)  
    # Remove instances
    df = remove_instances(df, OCT_OCTA_exclusion_criteria, OCTA_filter, OCT_filter)     # df.shape = 763x135
    # Remove rows (empty users > why happening?)
    df = remove_irrelevant_users(df)
    # Mean normalization
    df = scale_numerical_attributes(df)
    
    # to be deleted after
    writer = pd.ExcelWriter('output_df_test.xlsx')
    df.to_excel(writer)
    writer.save()

    eyes = '2e'
    if just_one_eye:
        # In case of use just one eye for patient
        df = select_one_eye(df)
        eyes = '1e'

    # Convert class values
    if classes == '0-1':
        y = df['CLASE'].copy()
        y[y > 1] = 1
        # y[y < 2] = 0 # here: added
        labels = [0,1] # here: maybe with [1,2] as ValueError with [0,1] : This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0
        print('Classes are:', classes)
        print('labels are:', labels)
    elif classes == '0-1-2':
        y = df['CLASE'].copy()
        y[y > 2] = 2
        labels = [0, 1, 2]
    elif classes == '1-2':
        df = df.drop(df[df.CLASE == 0].index)
        y = df['CLASE'].copy()
        y[y > 2] = 2
        labels = [1, 2]
    else:
        print('ERROR CLASS')


    path = os.path.join('..', 'results_' + str(labels[0]) + '_' + str(labels[1]) + '_' + eyes)
    if not os.path.exists(path):
        os.makedirs(path)

    loop_evaluation(test, features_vec, models, num_features, df, y, labels=labels,
                    save_cr=save_cr, save_cm=save_cm, save_rc=save_rc, eyes=eyes)

