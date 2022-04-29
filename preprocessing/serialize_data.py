from utils import *
from preprocessing.feature_extract import extract_mel_band_energies
import os


def prepare_datasets(cfg):
    """ Create csv files for training and testing datasets with folds 

    :param cfg: Configuration dictionary
    """
    df = create_csv(cfg['class_names'], cfg['paths']['audio_path'])

    # Create 10 folds
    df = create_folds(df, cfg['folds']['num_folds'])

    # 1 fold for testing
    test_df = df[df['fold'] == cfg['folds']['fold_for_testing']].drop(columns=[
                                                                      'fold'])

    # 9 folds for training
    train_df = df[df['fold'] != cfg['folds']['fold_for_testing']].drop(
        columns=['fold']).reset_index(drop=True)
    train_df = create_folds(train_df, cfg['folds']['train_folds'])

    # Add target column to dataframe as one hot encoding
    test_df = add_targets_to_df(test_df)
    train_df = add_targets_to_df(train_df)

    # Save the files
    test_df.to_csv(cfg['paths']['test_csv_path'], index=False)
    train_df.to_csv(cfg['paths']['train_csv_path'], index=False)


def serialize_data(cfg):
    """ Serialize the data into pickle files 

    :param cfg: Configuration dictionary
    """
    prepare_datasets(cfg)
    modes = ['train', 'test']
    print('Serializing datasets...')

    for mode in modes:
        df = load_csv(cfg['paths']['root_path'] + mode + '_metadata.csv')
        features_and_classes = {}
        df['feature_path'] = -1

        for i in range(len(df)):
            audio, _ = get_audio_file_data(df.iloc[i]['audio_path'])
            file_name = df.iloc[i]['audio_path'].split(
                '\\')[-1].replace('.au', '.pickle')
            MBE = extract_mel_band_energies(audio)
            features_and_classes.update(
                {'features': MBE, 'class': df.iloc[i]['target']})

            if not os.path.exists(cfg['paths']['root_path'] + mode + '_features'):
                os.mkdir(cfg['paths']['root_path'] + mode + '_features')

            feature_path = cfg['paths']['root_path'] + \
                mode + '_features/' + file_name
            df.loc[i, 'feature_path'] = feature_path
            serialize_features_and_classes(features_and_classes, feature_path)

        # Update csv and save
        df = df.drop(columns=['target'])
        df = df.to_csv(cfg['paths']['root_path'] +
                       mode + '_metadata.csv', index=False)

    print('Done!')


def serialize_split_data(cfg):
    train_df = load_csv(cfg['paths']['root_path'] + 'train_metadata.csv')
    # Add target column to dataframe as one hot encoding
    train_df = add_targets_to_df(train_df)
    column_names = ['audio_path', 'feature_path', 'target', 'fold']
    split_df = pd.DataFrame(columns=column_names)
    root_path = 'data\\split_audio'
    features_and_classes = {}
    for i in range(len(train_df)):
        for split in range(6):
            audio_path = train_df.iloc[i]['audio_path']
            class_name = train_df.iloc[i]['class_name']
            file_number = audio_path.split('\\')[-1].split('.')[1]
            file_path = root_path + '\\' + class_name + '\\' + class_name + \
                '.' + file_number + '_' + str(split) + '.wav'
                
            if not os.path.exists(file_path):
                continue
            feature_path = cfg['paths']['root_path'] +'train_features//' +  class_name + \
                '.' + file_number + '_' + str(split) + '.pickle'
            
            audio, _ = get_audio_file_data(file_path)
            MBE = extract_mel_band_energies(audio)
            features_and_classes.update(
                {'features': MBE, 'class': train_df.iloc[i]['target']})
            
            if not os.path.exists(cfg['paths']['root_path'] + 'train_features'):
                os.mkdir(cfg['paths']['root_path'] +'train_features')
            
            split_df = split_df.append({'audio_path': file_path, 'feature_path': feature_path, 'target': train_df.iloc[i]['target'], 'fold': train_df.iloc[i]['fold']}, ignore_index=True)
            #split_df.loc[i, 'feature_path'] = feature_path
            #split_df.loc[i, 'audio_path'] = file_path
            #split_df.loc[i, 'target'] = train_df.iloc[i]['target']
            #split_df.loc[i, 'fold'] = train_df.iloc[i]['fold']
            serialize_features_and_classes(features_and_classes, feature_path)
        
    
        
    # Update csv and save
    split_df = split_df.drop(columns=['target'])
    split_df = split_df.to_csv(cfg['paths']['root_path'] +
                    'split_metadata.csv', index=False)