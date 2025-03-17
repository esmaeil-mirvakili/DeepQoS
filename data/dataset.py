import json
import os.path
from ctypes import ArgumentError

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class IODataSet(Dataset):
    def __init__(self, path, stage='train', val_size=0, train_size=0.8, shuffle=False, seed=12,
                 exclude_normalization=None):
        if stage not in ['train', 'val', 'test']:
            raise ArgumentError(f'Unknown stage {stage}.')
        self.path = path
        self.stage = stage
        self.val_size = val_size
        self.train_size = train_size
        self.shuffle = shuffle
        self.seed = seed
        self.entries = None
        self.ops = None
        self.data = None
        self.labels = None
        self.len = 0
        self.op_types = None
        self.entry_types = None
        self.entry_log_transform_features = ['cost', 'latency']
        self.entry_standard_scale_features = ['cost', 'latency', 'ops_len']
        self.entry_minmax_scale_features = ['priority']
        self.ops_log_transform_features = ['len']
        self.ops_standard_scale_features = ['len']
        self.ops_minmax_scale_features = ['off']
        if exclude_normalization is None:
            exclude_normalization = []
        for exclude_column in exclude_normalization:
            if exclude_column in self.entry_log_transform_features:
                self.entry_log_transform_features.remove(exclude_column)
            if exclude_column in self.entry_standard_scale_features:
                self.entry_standard_scale_features.remove(exclude_column)
            if exclude_column in self.entry_minmax_scale_features:
                self.entry_minmax_scale_features.remove(exclude_column)
            if exclude_column in self.ops_log_transform_features:
                self.ops_log_transform_features.remove(exclude_column)
            if exclude_column in self.ops_standard_scale_features:
                self.ops_standard_scale_features.remove(exclude_column)
            if exclude_column in self.ops_minmax_scale_features:
                self.ops_minmax_scale_features.remove(exclude_column)
        self.load_data()
        self.preprocess()
        self.separate_labels()

    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)

    def input_size(self):
        return len(self.data.columns)

    def load_data(self):
        entries_path = os.path.join(self.path, 'entries.csv')
        self.entries = pd.read_csv(entries_path)
        ops_path = os.path.join(self.path, 'ops.csv')
        self.ops = pd.read_csv(ops_path)
        entry_type_path = os.path.join(self.path, 'msg_op_types.json')
        with open(entry_type_path, "r") as file:
            self.entry_types = json.load(file)
        op_type_path = os.path.join(self.path, 'osd_op_types.json')
        with open(op_type_path, "r") as file:
            self.op_types = json.load(file)

    # Function to apply log transformation
    @staticmethod
    def apply_log_transform(df, columns):
        for col in columns:
            df[col] = np.log1p(df[col])  # log(value + 1) to handle zeros
        return df

    # Function to apply standard scaling
    @staticmethod
    def apply_standard_scaling(df, columns):
        mean = df[columns].mean()
        std = df[columns].std()
        df[columns] = (df[columns] - mean) / std  # Standardization formula
        return df

    # Function to apply MinMax scaling
    @staticmethod
    def apply_minmax_scaling(df, columns):
        min_vals = df[columns].min()
        max_vals = df[columns].max()
        df[columns] = (df[columns] - min_vals) / (max_vals - min_vals)  # MinMax Scaling formula
        return df

    def preprocess(self):
        # mean = self.entries['latency'].mean()
        # print(mean)
        self.apply_log_transform(self.entries, self.entry_log_transform_features)
        self.apply_standard_scaling(self.entries, self.entry_standard_scale_features)
        self.apply_minmax_scaling(self.entries, self.entry_minmax_scale_features)
        self.apply_log_transform(self.ops, self.ops_log_transform_features)
        self.apply_standard_scaling(self.ops, self.ops_standard_scale_features)
        self.apply_minmax_scaling(self.ops, self.ops_minmax_scale_features)

        all_io_types = list(range(len(self.op_types)))

        # Count number of operations per io_type per index
        io_counts = self.ops.groupby(['index', 'type']).size().unstack(fill_value=0)
        io_counts = io_counts.reindex(columns=all_io_types, fill_value=0)  # Ensure all 81 columns exist
        io_counts.columns = [f'io_type_{col}_num' for col in io_counts.columns]

        # Aggregate sum of len and mean of offset per io_type per index
        io_agg = self.ops.groupby(['index', 'type']).agg(
            sum_len=('len', 'sum'),
            mean_offset=('off', 'mean')
        ).unstack(fill_value=0)

        # Ensure all io_types are represented
        extra_io_agg = io_agg.reindex(
            columns=pd.MultiIndex.from_product([['sum_len', 'mean_offset'], all_io_types], names=['metric', 'io_type']),
            fill_value=0)

        # Flatten MultiIndex columns correctly
        extra_io_agg.columns = [f'{col[0]}_io_type_{col[1]}' for col in extra_io_agg.columns]
        extra_io_agg.reset_index(inplace=True)

        all_entry_categories = list(range(len(self.entry_types)))  # Ensure all categories from 0 to 100 are included

        # One-hot encode 'x'
        onehot_encoder = OneHotEncoder(sparse_output=False, categories=[all_entry_categories], handle_unknown='ignore')
        entry_type_encoded = onehot_encoder.fit_transform(self.entries[['type']])

        # Convert to DataFrame
        type_encoded_df = pd.DataFrame(entry_type_encoded, columns=[f'req_type_{i}' for i in all_entry_categories])

        # Merge back with original df
        self.entries = pd.concat([self.entries, type_encoded_df], axis=1)
        self.entries.drop(columns=['type'], inplace=True)
        self.entries.drop(columns=['data_len'], inplace=True)
        self.entries.drop(columns=['data_off'], inplace=True)
        self.entries.drop(columns=['dequeue_end_stamp'], inplace=True)
        self.entries.drop(columns=['dequeue_stamp'], inplace=True)
        self.entries.drop(columns=['enqueue_stamp'], inplace=True)
        self.entries.drop(columns=['recv_stamp'], inplace=True)
        self.entries.drop(columns=['owner'], inplace=True)

        # Merge aggregated features with request dataset
        self.data = self.entries.merge(io_counts, on='index', how='left').fillna(0)
        self.data = self.data.merge(extra_io_agg, on='index', how='left').fillna(0)
        self.data.drop(columns=['index'], inplace=True)
        self.data.sort_values(by="timestamp", inplace=True)
        self.data.drop(columns=['timestamp'], inplace=True)
        train_end = int(len(self.data) * self.train_size)
        val_end = train_end + int(len(self.data) * self.val_size)
        if self.stage == 'train':
            self.data = self.data.iloc[:train_end]
        elif self.stage == 'val':
            self.data = self.data.iloc[train_end:val_end]
        else:
            self.data = self.data.iloc[:val_end]

    def separate_labels(self):
        self.labels = self.data['latency']
        self.data.drop(columns=['latency'], inplace=True)

    def __getitem__(self, idx):
        features = self.data.iloc[idx].to_numpy()
        label = self.labels.iloc[idx]
        features = features.astype(np.float32)
        label = label.astype(np.int64)
        return features, label


class IOBinClassificationDataSet(IODataSet):
    def __init__(self, path, stage='train', val_size=0, train_size=0.8, shuffle=False, seed=12, threshold=2_000_000):
        self.threshold = threshold
        super(IOBinClassificationDataSet, self).__init__(path, stage=stage, val_size=val_size, train_size=train_size,
                                                         shuffle=shuffle, seed=seed, exclude_normalization=['latency'])

    def preprocess(self):
        super().preprocess()
        self.data['latency'] = (self.data['latency'] >= self.threshold).astype(int)
