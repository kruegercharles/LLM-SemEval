from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value, Sequence, ClassLabel, load_from_disk
import pandas as pd
import re
import os

DATASET_IDS = {
    'off' : [
        os.path.join(os.path.dirname(__file__), '../../data/codabench_data/train/eng_a.csv')
    ],
    'on' : [
        'SkyWater21/lt_go_emotions',
        #'SkyWater21/ru_emotions'
    ]
}

LABEL_MAPPING_CODABENCH = {
    0 : 'anger',
    1 : 'fear',
    2 : 'joy',
    3 : 'sadness', 
    4 : 'surprise',
    5 : 'neutral',
}

LABEL_MAPPING_EKMAN = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'joy',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral',
}


def retrieve_datasets():
    """
    Uses the specified datapath to compile a single large dataset. 

    Params:
        None 
    Returns:
        datasets.Dataset - A dataset object.
    """
    
    dsets = []
    for dset_key in DATASET_IDS.keys():
        if dset_key == 'off':
            for off_dset in DATASET_IDS['off']:
                ds = retrieve_offline_data(off_dset)
                features = Features({'text': Value(dtype='string', id=None), 'labels': Sequence(feature=ClassLabel(names=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral'], id=None), length=-1, id=None)})
                dsets.append(Dataset.from_dict({'text' : ds.keys(), 'labels' : ds.values()}, features=features))
        elif dset_key == 'on':
            for on_dset in DATASET_IDS['on']:
                print(f"Now processing dataset: {on_dset}")
                dsets.append(retrieve_online_data(on_dset))
    return concatenate_datasets(dsets)



def retrieve_offline_data(file_id : str):
    """
    Processes data from a local path.

    Params:
        file_id : The ID (path) of the local data file.
    Returns: 
        df_dict : A dictionary containing the preprocessed data entries (text & label).
    """
    # read in the file
    df = pd.read_csv(file_id, skiprows=1, header=None)
    # drops the first column (ids)
    df.drop(df.columns[0], axis=1, inplace=True)
    # normalizes the text of the second column
    df.iloc[:, 0] = df.iloc[:, 0].str.lower().replace(r'[^\w\s]', '', regex=True)
    # converts last two columns into a single column indocating the label [0-4]
    df['labels'] = df[df.columns[-5:]].apply(lambda row: [i - len(df.columns) + 5 for i, x in enumerate(row, start=len(df.columns) - 5) if x == 1], axis=1)
    # if no category is present apply the label 5 = 'neutral' to the sample
    df['labels'] = df['labels'].apply(lambda x: [label + 1 if label != 0 else label for label in x] if x else [6])
    # drop original label columns
    df.drop(df.columns[-6:-1], axis=1, inplace=True)
    # reorder the dataframe so that 'text' comes first and 'label' second
    df = df[[df.columns[0], 'labels']]
    # convert string lists to integer lists
    df['labels'] = df['labels'].apply(lambda x: list(map(int, x)) if isinstance(x, list) else [x])
    # convert dataset to dict so that it can be compiled to a dataset format
    df_dict = pd.Series(df.labels.values, index=df.iloc[:, 0]).to_dict()
    #print(df_dict)
    #df.to_csv('/Users/eliaruhle/Documents/LLM-SemEval/data/codabench_data/train/pre_eng_a.csv', index=False)
    #print(df.head())
    return df_dict
    

def retrieve_online_data(file_id : str):
    """
    Processes data from an online source.

    Params:
        file_id : The ID (path) of the data file.
    Returns: 
        ds : A dataset.Dataset containing preprocessed data entries (text & label).
    """

    if isinstance(file_id, list) and len(file_id) > 1:
        ds = load_dataset(file_id[0], file_id[1])
    else:
        try:
            ds = load_dataset(file_id)['train']
        except:
            print('Searching for comb_train attribute.')
            try:
                ds = load_dataset(file_id)['comb_train']
            except:
                print('Only loading train data not supported! \n Forcing default download.')
                ds = load_dataset(file_id)
    
    for col in ds.column_names:
        if col not in ['text', 'labels']:
            ds = ds.remove_columns([col])
        if col == 'text':
            ds = ds.map(lambda x: {col : re.sub(r'[^\w\s]', '', x[col].lower())})
    return ds

def one_hot_encoding(dataset : Dataset, num_classes : int):
    """
    Transforms the 'labels' column of a Dataset object into one-hot encoded vectors.

    Args:
        dataset (Dataset): A Dataset object.
        num_classes (int): Number of classes. (Used regarding emotion scheme)

    Returns:
        Dataset: The input dataset with one-hot encoded labels.
    """
    one_hot_encoded_labels = list()

    for labels in dataset['labels']:
        one_hot = [0] * num_classes

        for label in labels:
            one_hot[label] = 1
        
        one_hot_encoded_labels.append(one_hot)
    
    # Replace the 'labels' column with the new one-hot encoded labels
    dataset = dataset.map(lambda x, idx: {**x, 'labels': one_hot_encoded_labels[idx]}, with_indices=True)

    return dataset