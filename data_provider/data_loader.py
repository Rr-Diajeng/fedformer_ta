import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    # def __read_data__(self):
    #     self.scaler = StandardScaler()
    #     df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

    #     if 'time' in df_raw.columns:
    #         df_raw['date'] = pd.to_datetime(df_raw['date'] + ' ' + df_raw['time'])
    #         df_raw = df_raw.drop(columns=['time'])
    #     else:
    #         df_raw['date'] = pd.to_datetime(df_raw['date'])

    #     '''
    #     df_raw.columns: ['date', ...(other features), target feature]
    #     '''
    #     cols = list(df_raw.columns)
    #     cols.remove(self.target)
    #     cols.remove('date')
    #     df_raw = df_raw[['date'] + cols + [self.target]]
    #     # print(cols)
    #     num_train = int(len(df_raw) * 0.7)
    #     num_test = int(len(df_raw) * 0.2)
    #     num_vali = len(df_raw) - num_train - num_test
    #     border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
    #     border2s = [num_train, num_train + num_vali, len(df_raw)]
    #     border1 = border1s[self.set_type]
    #     border2 = border2s[self.set_type]

    #     if self.features == 'M' or self.features == 'MS':
    #         cols_data = df_raw.columns[1:]
    #         df_data = df_raw[cols_data]
    #     elif self.features == 'S':
    #         df_data = df_raw[[self.target]]

    #     if self.scale:
    #         train_data = df_data[border1s[0]:border2s[0]]
    #         self.scaler.fit(train_data.values)
    #         data = self.scaler.transform(df_data.values)
    #     else:
    #         data = df_data.values

    #     df_stamp = df_raw[['date']][border1:border2]
    #     df_stamp['date'] = pd.to_datetime(df_stamp.date)
    #     if self.timeenc == 0:
    #         df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    #         df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    #         df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    #         df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    #         data_stamp = df_stamp.drop(['date'], 1).values
    #     elif self.timeenc == 1:
    #         data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
    #         data_stamp = data_stamp.transpose(1, 0)

    #     self.data_x = data[border1:border2]
    #     self.data_y = data[border1:border2]
    #     self.data_stamp = data_stamp

    def __read_data__(self):
        self.scaler = StandardScaler()
        # 1. Baca & parse tanggal
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if 'time' in df_raw.columns:
            df_raw['date'] = pd.to_datetime(df_raw['date'] + ' ' + df_raw['time'])
            df_raw = df_raw.drop(columns=['time'])
        else:
            df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        df_raw['month'] = df_raw['date'].dt.month

        # 2. Filter per flag (tambah buffer untuk sliding window)
        if self.set_type == 0:       # train
            df_sel = df_raw[df_raw['month'].isin([5,6,7])]

        elif self.set_type == 1:     # val
            df_temp = df_raw[df_raw['month'].isin([7,8])]
            val_start_idx = df_temp[df_temp['month'] == 8].index[0]
            start_idx = val_start_idx - self.seq_len
            df_sel = df_temp.loc[start_idx:]

        else:                        # test
            df_temp = df_raw[df_raw['month'].isin([8,9])]
            test_start_idx = df_temp[df_temp['month'] == 9].index[0]
            start_idx = test_start_idx - self.seq_len
            df_sel = df_temp.loc[start_idx:]

        print(f"{['Train','Val','Test'][self.set_type]} | Start: {df_sel['date'].iloc[0]} | End: {df_sel['date'].iloc[-1]} | Total: {len(df_sel)}")

        # 3. Siapkan fitur
        # daftar kolom fitur kecuali date, target, month
        feature_cols = [c for c in df_sel.columns if c not in ['date', self.target, 'month']]
        if self.features in ['M','MS']:
            data_all   = df_sel[feature_cols + [self.target]].values
            train_part = df_raw[df_raw['month'].isin([5,6,7])][feature_cols + [self.target]]
        else:  # 'S'
            data_all   = df_sel[[self.target]].values
            train_part = df_raw[df_raw['month'].isin([5,6,7])][[self.target]]

        # 4. Scaling (fit hanya di train months)
        if self.scale:
            self.scaler.fit(train_part.values)
            data = self.scaler.transform(data_all)
        else:
            data = data_all

        # 5. Timestamp features
        df_stamp = df_sel[['date']].copy()
        if self.timeenc == 0:
            df_stamp['month']   = df_stamp.date.dt.month
            df_stamp['day']     = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour']    = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop('date', axis=1).values
        else:
            data_stamp = time_features(pd.DatetimeIndex(df_stamp['date']), freq=self.freq).T

        # 6. Assign
        self.data_x     = data
        self.data_y     = data
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# class Dataset_Pred(Dataset):
#     def __init__(self, root_path, flag='pred', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['pred']

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cols = cols
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         if self.cols:
#             cols = self.cols.copy()
#             cols.remove(self.target)
#         else:
#             cols = list(df_raw.columns)
#             cols.remove(self.target)
#             cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         border1 = len(df_raw) - self.seq_len
#         border2 = len(df_raw)

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             self.scaler.fit(df_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         tmp_stamp = df_raw[['date']][border1:border2]
#         tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
#         pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

#         df_stamp = pd.DataFrame(columns=['date'])
#         df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         if self.inverse:
#             self.data_y = df_data.values[border1:border2]
#         else:
#             self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         if self.inverse:
#             seq_y = self.data_x[r_begin:r_begin + self.label_len]
#         else:
#             seq_y = self.data_y[r_begin:r_begin + self.label_len]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(
        self,
        root_path,
        size=None,                # [seq_len, label_len, pred_len]
        features='S',
        data_path='data.csv',
        target='OT',
        scale=True,
        inverse=False,
        timeenc=0,
        freq='h',
        pred_range=None          # <-- NEW: e.g. ('2020-01-01 00:00:00','2020-12-31 23:00:00')
    ):
        # 1) setup lengths
        if size is None:
            self.seq_len   = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len  = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        # 2) store everything
        self.root_path  = root_path
        self.data_path  = data_path
        self.features   = features
        self.target     = target
        self.scale      = scale
        self.inverse    = inverse
        self.timeenc    = timeenc
        self.freq       = freq
        self.pred_range = pred_range

        # 3) build
        self.__read_data__()

    def __read_data__(self):
        # — load & parse date
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if 'time' in df_raw.columns:
            df_raw['date'] = pd.to_datetime(df_raw['date'] + ' ' + df_raw['time'])
            df_raw.drop(columns=['time'], inplace=True)
        else:
            df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw['month'] = df_raw['date'].dt.month

        # — select features exactly like your Custom dataset
        feat_cols = [c for c in df_raw.columns if c not in ['date', self.target, 'month']]
        if self.features in ['M', 'MS']:
            data_all   = df_raw[feat_cols + [self.target]].values
            train_part = df_raw[df_raw['month'].isin([5,6,7])][feat_cols + [self.target]].values
        else:
            data_all   = df_raw[[self.target]].values
            train_part = df_raw[df_raw['month'].isin([5,6,7])][[self.target]].values

        # — scale on train months
        if self.scale:
            self.scaler = StandardScaler().fit(train_part)
            data = self.scaler.transform(data_all)
        else:
            self.scaler = None
            data = data_all

        # — timestamp features for **all** dates
        df_stamp = pd.DataFrame({'date': df_raw['date']})
        if self.timeenc == 0:
            df_stamp['month']   = df_stamp.date.dt.month
            df_stamp['day']     = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour']    = df_stamp.date.dt.hour
            if 'min' in self.freq:
                bucket = int(self.freq.replace('min',''))
                df_stamp['minute'] = df_stamp.date.dt.minute // bucket
            stamp_all = df_stamp.drop(columns=['date']).values
        else:
            tf = time_features(pd.DatetimeIndex(df_stamp['date']), freq=self.freq)
            stamp_all = tf.T

        # — carve out the **last** seq_len for inputs
        border1 = len(data) - self.seq_len
        border2 = len(data)
        self.data_x = data[border1:border2]
        if self.inverse:
            raw_vals   = (df_raw[self.target].values.reshape(-1,1)
                          if self.features=='S'
                          else df_raw[feat_cols + [self.target]].values)
            self.data_y = raw_vals[border1:border2]
        else:
            self.data_y = data[border1:border2]

        # — FUTURE INDEX: either your pred_range or rolling default
        last_date = df_raw['date'].iloc[-1]
        if self.pred_range:
            start, end = self.pred_range
            future_idx  = pd.date_range(start, end, freq=self.freq)
            self.pred_len = len(future_idx)
        else:
            future_idx = pd.date_range(last_date, periods=self.pred_len+1, freq=self.freq)[1:]

        # — extend the stamps
        hist_stamp = df_stamp.iloc[border1:border2][['date']].copy()
        future_df  = pd.DataFrame({'date': future_idx})
        df_ext     = pd.concat([hist_stamp, future_df], ignore_index=True)

        if self.timeenc == 0:
            df_ext['month']   = df_ext.date.dt.month
            df_ext['day']     = df_ext.date.dt.day
            df_ext['weekday'] = df_ext.date.dt.weekday
            df_ext['hour']    = df_ext.date.dt.hour
            if 'min' in self.freq:
                bucket = int(self.freq.replace('min',''))
                df_ext['minute'] = df_ext.date.dt.minute // bucket
            self.data_stamp = df_ext.drop(columns=['date']).values
        else:
            tf2 = time_features(pd.DatetimeIndex(df_ext['date']), freq=self.freq)
            self.data_stamp = tf2.T

    def __getitem__(self, idx):
        # we only have one window, so idx is always 0
        s_begin = 0
        s_end   = self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        seq_x      = self.data_x[s_begin:s_end]
        seq_y      = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return 1

    def inverse_transform(self, data):
        return (self.scaler.inverse_transform(data)
                if self.scaler is not None else data)
