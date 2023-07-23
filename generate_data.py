import os
import argparse
import numpy as np
import pandas as pd
import math
def generate_data(args, path_input, path_output):
    if 'txt' in path_input or 'csv' in path_input:
        file = open(path_input)
        rawdata = np.loadtxt(file, delimiter=',')
        times, num_nodes = rawdata.shape # (times, num_nodes)
        print("times: ", times, ", num_nodes: ", num_nodes)
    elif 'npz' in path_input:
        file = np.load(path_input)
        if len(file['data'].shape) == 3:
            if args.dataset == 'stock':
                rawdata = np.squeeze(file['data'][:, :, 1])
            else:
                rawdata = np.squeeze(file['data'][:, :, 0])
        else:
            rawdata = np.squeeze(file['data'])
        if len(rawdata.shape)==2:
            times, num_nodes = rawdata.shape
            print("times: ", times, ", num_nodes: ", num_nodes)
        else:
            times, num_nodes, features_dim = rawdata.shape
            print("times: ", times, ", num_nodes: ", num_nodes, ", features_dim: ", features_dim)
    elif 'npy' in path_input:
        file = np.load(path_input)
        if len(file.shape) == 3:
            if args.dataset == 'stock':
                rawdata = np.squeeze(file[:, :, 1])
            else:
                rawdata = np.squeeze(file[:, :, 0])
        else:
            rawdata = np.squeeze(file)
        if len(rawdata.shape)==2:
            times, num_nodes = rawdata.shape
            print("times: ", times, ", num_nodes: ", num_nodes)
        else:
            times, num_nodes, features_dim = rawdata.shape
            print("times: ", times, ", num_nodes: ", num_nodes, ", features_dim: ", features_dim)

    P = args.window
    h = args.horizon

    train_end = int(times * args.train_rate)
    val_end = int(times * (args.train_rate + args.val_rate))
    train_set = range(0, train_end)
    val_set = range(train_end, val_end)
    test_set = range(val_end, times)

    # train
    x_train, y_train = split_x_y(rawdata, train_set, P, h)
    # val
    x_val, y_val = split_x_y(rawdata, val_set, P, h)
    # test
    x_test, y_test = split_x_y(rawdata, test_set, P, h)

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    # Write the data into npz file.
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(os.path.join(path_output, "%s.npz" % cat), x=_x, y=_y)

def split_x_y(rawdata, idx_set, P, h):
    x, y = [], []
    samples = len(idx_set) - P - h + 1
    for i in range(samples):
        start = idx_set[i]
        endx = start + P
        endy = endx + h
        x.append(rawdata[start:endx,...])
        y.append(rawdata[endx:endy,...])
    x = np.stack(x, axis=0) # (samples, P, num_nodes)
    y = np.stack(y, axis=0) # (samples, h, num_nodes)

    if len(x.shape)==3:
        return np.expand_dims(x, axis = -1), np.expand_dims(y, axis = -1)
    else:
        return x, y

def generate_data_npz(args, path_input, path_output):
    if 'txt' in path_input or 'csv' in path_input:
        file = open(path_input)
        rawdata = np.loadtxt(file, delimiter=',')
        df = pd.DataFrame(rawdata)
        times, num_nodes = rawdata.shape  # (times, num_nodes)
        print("times: ", times, ", num_nodes: ", num_nodes)
    elif 'npz' in path_input:
        file = np.load(path_input)
        if len(file['data'].shape) == 3:
            rawdata = np.squeeze(file['data'][:,:,0])
        else:
            rawdata = np.squeeze(file['data'])
        print(rawdata.shape)
        df = pd.DataFrame(rawdata)
        print("df:{}".format(type(df)))
    x_offsets = np.arange(-11, 1, 1)  # array([-11,-10,...,0])
    y_offsets = np.arange(1, 13, 1)  # array([1,2,...,12])

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_npz_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=args.add_time_in_day,
        add_day_in_week=args.add_day_in_week,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # Write the data into npz file.
    # train/val/test: 7/1/2
    num_samples = x.shape[0]
    num_train = round(num_samples * args.train_rate)
    num_val = round(num_samples * args.val_rate)
    num_test = num_samples - num_train - num_val

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(path_output, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1])
        )


def generate_npz_seq2seq_io_data(df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None):
    num_samples, num_nodes = df.shape  # (times, num_nodes)
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:  # True
        # df.index.values = np.array(df.index.values)
        time_ind = np.array(df.index.values % 288 ) / np.array([288])
        print("time_ind:{}".format(time_ind))
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        # (1,1,times)->copy->(1,num_nodes,times)->transpose->(times,num_nodes,1)
        data_list.append(time_in_day)
    if add_day_in_week:  # False
        dayofweek = np.zeros(df.shape[0], dtype=int)
        week_time = 5*12*24*7
        day_time = 5*12*24
        summ = 0
        for i in range(df.shape[0]):
            summ += 5
            dayofweek[i] = int((summ%week_time)//day_time)
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))  # (times,num_nodes,7)
        day_in_week[np.arange(num_samples), :, dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)  # (times,num_nodes,2)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - max(y_offsets))
    for t in range(min_t, max_t):  # times-11-12 = samples
        x_t = data[t + x_offsets, ...]  # (12,num_nodes,2)
        y_t = data[t + y_offsets, ...]  # (12,num_nodes,2)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)  # x,y: (samples,12,num_nodes,2)
    return x, y

def generate_data_h5(args, path_input, path_output):
    df = pd.read_hdf(path_input)
    x_offsets = np.arange(-11, 1, 1) # array([-11,-10,...,0])
    y_offsets = np.arange(1, 13, 1) # array([1,2,...,12])
    
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=args.add_time_in_day,
        add_day_in_week=args.add_day_in_week,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    
    # Write the data into npz file.
    # train/val/test: 7/1/2
    num_samples = x.shape[0]
    num_train = round(num_samples * args.train_rate)
    num_val = round(num_samples * args.val_rate)
    num_test = num_samples - num_train - num_val

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(path_output, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1])
        )

def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None):
    num_samples, num_nodes = df.shape # (times, num_nodes)
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day: # True
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)) 
        # (1,1,times)->copy->(1,num_nodes,times)->transpose->(times,num_nodes,1)
        data_list.append(time_in_day)
    if add_day_in_week: # False
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))    # (times,num_nodes,7)
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)
    
    data = np.concatenate(data_list, axis=-1) # (times,num_nodes,2)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - max(y_offsets))
    for t in range(min_t, max_t): # times-11-12 = samples
        x_t = data[t + x_offsets, ...] # (12,num_nodes,2)
        y_t = data[t + y_offsets, ...] # (12,num_nodes,2)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0) # x,y: (samples,12,num_nodes,2)
    return x, y
        
def main(args):
    print("Generating training data:")
    if args.dataset == "Solar_AL":
        print("Solar_AL:")
        generate_data(args, "./data/solar_AL/solar_AL.txt", "./data/solar_AL/")
    if args.dataset == "PEMS04":
        print("PEMS04:")
        generate_data_npz(args, "./data/PEMS04/pems04.npz", "./data/PEMS04/")
    if args.dataset == "PEMS08":
        print("PEMS08:")
        generate_data_npz(args, "./data/PEMS08/pems08.npz", "./data/PEMS08/")
    if args.dataset == "PEMS03":
        print("PEMS03")
        generate_data(args, "./data/PEMS03/PEMS03.npz", "./data/PEMS03/")
    if args.dataset == "PEMS07":
        print("PEMS07")
        generate_data(args, "./data/PEMS07/PEMS07.npz", "./data/PEMS07/")
    if args.dataset == "stock":
        print("stock")
        generate_data(args, "./data/stock/stock.npz","./data/stock")
    print("Finish!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train_rate", type=float, default=0.7)
    parser.add_argument("--val_rate", type=float, default=0.1)
    parser.add_argument("--add_time_in_day", type=bool, default=True)
    parser.add_argument("--add_day_in_week", type=bool, default=True)
    parser.add_argument("--dataset",type=str, default="Solar_AL")
    args = parser.parse_args()
    main(args)