import os
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WellCorrelationDataset:

    def __init__(self,
                 root : str = None,
                 train_preload : bool = False,
                 test_preload : bool = False,
                 fill_na_method : str = None):
        self.root : str = root
        self.train_path : str = os.path.join(self.root, 'train.csv')
        self.test_path : str = os.path.join(self.root, 'test.csv')
        self.train_preload : bool = train_preload
        self.test_preload : bool = test_preload
        self.fill_na_method : str = fill_na_method
        self.train_data : pd.DataFrame = None
        self.test_data : pd.DataFrame = None
        self.wells_list : List[str] = None
        self.target_groups = None
        self.group_num_map = {}
        self.used_cols = ['WELL', 'Z_LOC', 'GR', 'DTC', 'GROUP']

        self.__preload__()

    def __preload__(self):
        if self.train_preload:
            self.train_data = pd.read_csv(self.train_path, delimiter=';')
            self.train_data.loc[:, 'Z_LOC'] *= -1
            self.train_data.sort_values(by = ['WELL', 'Z_LOC'])
            self.wells_list = self.train_data.WELL.unique()
            self.target_groups = self.train_data.GROUP.unique()
            for i, label in enumerate(self.target_groups):
                self.group_num_map.update({label : i})
            self.train_data.loc[:, 'GROUP_NUM'] = self.train_data.GROUP.map(self.group_num_map)
            self.used_cols += ['GROUP_NUM']

        if self.test_preload:
            self.test_preload = pd.read_csv(self.test_path, delimiter=';')

    def get_well_data(self, idx : str):
        assert(idx in self.wells_list)
        well_df = self.train_data.loc[self.train_data.WELL == idx, self.used_cols]
        if self.fill_na_method is None:
            well_df.fillna(0, inplace = True)
        else:
            assert(self.fill_na_method in ['ffill', 'bfill', 'pad'])
            well_df.fillna(method = self.fill_na_method, inplace = True)

        well_df.dropna(inplace = True)
        return well_df

    def get_matching_wells_id(self, tmpt : str = ""):
        m_wells = []
        for w in self.wells_list:
            m_wells += [w] if tmpt in w else []
        return m_wells

    def get_random_well_data(self, max_itr : int = 10, n_thres : int = 100) -> pd.DataFrame:
        for _ in range(max_itr):
            well_idx = np.random.randint(0, len(self.wells_list))
            well_idx = self.wells_list[well_idx]
            data = self.get_well_data(well_idx)
            if data.shape[0] >= n_thres:
                return data

        return None

class EncoderDecoderStaticDataset(Dataset):

    def __init__(self, root : str = None,
                 patch_size : int = 100,
                 resample : bool = True,
                 features : List[str] = ['GR', 'DTC'],
                 fill_na_method : str = None,
                 max_itr : int = 10,
                 train : bool = True,
                 test : bool = False) -> None:
        self.root = root
        self.patch_size = patch_size
        self.resample = resample
        self.features = features
        self.max_itr = max_itr

        self.dataset : WellCorrelationDataset = WellCorrelationDataset(self.root,
                                                                       fill_na_method=fill_na_method,
                                                                       train_preload=train,
                                                                       test_preload=test)

    def __getitem_random__(self) -> Any:

        df = self.dataset.get_random_well_data(self.max_itr, self.patch_size)
        assert(not df is None)

        for col in self.features:
            assert(col in df.columns)

        data = df.loc[:, self.features].values
        assert(self.patch_size <= len(data))

        i = np.random.randint(0, len(data)-self.patch_size)
        data_patch = data[i:i+self.patch_size, :]

        return data_patch

    def __getitem__(self, index: Any) -> Any:
        df = self.dataset.get_well_data(self.dataset.wells_list[index])

        for col in self.features:
            assert(col in df.columns)

        data = df.loc[:, self.features].values
        assert(self.patch_size <= len(data))

        i = np.random.randint(0, len(data)-self.patch_size)
        data_patch = torch.from_numpy(np.array(data[i:i+self.patch_size, :], dtype=np.float32))

        data_patch /= torch.norm(data_patch, dim=0, keepdim=True)+0.0000001
        data_patch /= torch.norm(data_patch, dim=1, keepdim=True)+0.0000001

        return data_patch.T

    def __len__(self):
        return self.dataset.wells_list.__len__()
