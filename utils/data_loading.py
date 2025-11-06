import os
import glob

import torch
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from pipeline_preprocessing import ApplyThreshold, _ConcatDataFrames, _SeparateDataFrames, CreateConcatDataset
from custom_datasets import sequential_dataset, unsupervised_sequential_dataset
from reproducibility_utils import seed_worker

feature_names = ["handwheelAngle", 
                 "throttle", 
                 "brake", 
                 "vxCG", 
                 "axCG", 
                 "ayCG", 
                 "yawRate", 
                 "rollRate", 
                 "pitchRate", 
                 "rollAngle", 
                 "pitchAngle", 
                 "chassisAccelFL", 
                 "chassisAccelFR", 
                 "chassisAccelRL", 
                 "chassisAccelRR", 
                 "wheelAccelFL", 
                 "wheelAccelFR", 
                 "wheelAccelRL", 
                 "wheelAccelRR", 
                 "deflectionFL", 
                 "deflectionFR", 
                 "deflectionRR"]

def load_data(
    train_path,
    train_ratio = None,
    columns_to_standardize = feature_names,
    columns_to_drop = [],
    sequence_length = 20,
    target = None,
    threshold = False,
    threshold_value = None,
    threshold_column = None,
    batchsize = 512,
    scaling = True,
    seed = 42,
    shuffle = True,
    ):
    """
    Load and preprocess data from csv files.

    Parameters:
    train_path (str): Path to the training data. folder that contains (multiple) csv's.
    train_ratio (float): Ratio of the training data to use.
    columns_to_standardize (list): List of features to standardize.
    columns_to_drop (list): List of features to drop.
    sequence_length (int): Length of the sequences.
    target (str): The target column.
    threshold (bool, optional): Whether to apply a threshold.
    treshold_value (float, optional): The threshold value.
    treshold_column (str, optional): The column to apply the threshold to.
    scaling (bool, optional): Whether to apply scaling. Defaults to True.
    seed (int, optional): Random seed. Defaults to 42.

    Returns:
    train_dataloader (DataLoader): DataLoader for the training data.
    val_dataloader (DataLoader): DataLoader for the validation data.
    """

    g = torch.Generator()
    g.manual_seed(seed)

    dfs = []

    for filename in sorted(glob.glob(os.path.join(train_path, '*.csv'))):
        print(filename)
        dfs.append(pd.read_csv(filename))

    ct = ColumnTransformer([("stand", StandardScaler(), columns_to_standardize)],
                       remainder="passthrough",
                       verbose_feature_names_out=False)

    pipeline = Pipeline([('threshold', ApplyThreshold(threshold=threshold_value, by=threshold_column, seq_length=sequence_length)), 
                        ('concat', _ConcatDataFrames()), 
                        ('stand', ct.set_output(transform="pandas")),
                        ('separate', _SeparateDataFrames())
                        ])
    
    # check if data for ssl or fine-tuning should be loaded
    if target == None:
        # ssl
        pipeline.steps.append(['concat dataset',CreateConcatDataset(unsupervised_sequential_dataset, seq_length=sequence_length, columns_to_drop=columns_to_drop)])
    else:
        # supervised
        pipeline.steps.append(['concat dataset',CreateConcatDataset(sequential_dataset, target=target, seq_length=sequence_length, columns_to_drop=columns_to_drop)])

    # check if a threshold should be applied
    if threshold == False:
         pipeline.steps = [step for step in pipeline.steps if step[0] not in ['threshold']]

    # check if scaling should be applied
    if scaling == False:
        pipeline.steps = [step for step in pipeline.steps if step[0] not in ['concat', 'stand', 'separate']]
         
    # run pipeline, split dataset into train and validation
    train_dataset = pipeline.fit_transform(dfs)

    if scaling == False:
        mean_dict = {}
        std_dict = {}
    else:
        mean_dict = {}
        std_dict = {}
        mean = ct.transformers_[0][1].mean_.tolist()
        std = ct.transformers_[0][1].scale_.tolist()
        for idx, col in enumerate(columns_to_standardize):
            mean_dict[col] = mean[idx]
            std_dict[col] = std[idx]
    
    if train_ratio is not None:
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_ratio, 1-train_ratio], generator=g) ##comment out
        val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)
    else:
        val_dataloader = None
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)
    
    return train_dataloader, val_dataloader, pipeline
    
def load_test_data(
    test_path,
    batchsize,
    pipeline,
    seed = 42,
    load_anonymized_data = False):

    g = torch.Generator()
    g.manual_seed(seed)
    
    dfs = []
    for filename in sorted(glob.glob(os.path.join(test_path, '*.csv'))):
        print(filename)
        df = pd.read_csv(filename)
        # Add missing columns with dummy values (they will be dropped by the pipeline)
        if load_anonymized_data:
            df['sideSlip'] = 0.0
            df['vyCG'] = 0.0
        dfs.append(df)

    test_dataset = pipeline.transform(dfs)
    
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, worker_init_fn=seed_worker, generator=g)
    return test_dataloader