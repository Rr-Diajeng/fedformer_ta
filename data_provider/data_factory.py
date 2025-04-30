from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    # default = Dataset_Custom
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last    = True
        batch_size   = args.batch_size
        freq         = args.freq

    elif flag == 'pred':
        # switch ke Dataset_Pred
        Data         = Dataset_Pred
        shuffle_flag = False
        drop_last    = False
        batch_size   = 1
        freq         = args.detail_freq

    else:  # train / val
        shuffle_flag = True
        drop_last    = True
        batch_size   = args.batch_size
        freq         = args.freq

    # common kwargs untuk kedua Dataset
    common_kwargs = dict(
        root_path = args.root_path,
        data_path = args.data_path,
        size      = [args.seq_len, args.label_len, args.pred_len],
        features  = args.features,
        target    = args.target,
        timeenc   = timeenc,
        freq      = freq,
        scale     = args.scale,
    )

    if flag == 'pred':
        # tambahkan argumen yang hanya untuk prediksi
        # misal args.pred_start="2020-01-01 00:00:00", args.pred_end="2020-12-31 23:00:00"
        data_set = Data(
            **common_kwargs,
            inverse    = args.inverse,      # kalau perlu output raw
            pred_range = (args.pred_start, args.pred_end)
        )
    else:
        # pass-through flag ke Dataset_Custom
        data_set = Data(
            **common_kwargs,
            flag = flag
        )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size  = batch_size,
        shuffle     = shuffle_flag,
        num_workers = args.num_workers,
        drop_last   = drop_last
    )
    return data_set, data_loader