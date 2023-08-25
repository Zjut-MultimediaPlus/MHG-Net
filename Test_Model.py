import math
import os
import os
from pandas import DataFrame
from scipy.stats import pearsonr, stats, spearmanr
import seaborn as sns
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import Set_seed
from MHG_Dataset import Dataset
import Config as config

transform_test = None

def pearson(x, y):
   return pearsonr(x, y)

def test_npys(dataset):

    model = torch.load(config.predict_model).to(config.device)

    model.eval()

    wind_error = 0
    RMW_error = 0
    pressure_error = 0

    wind_bias_sum = 0
    RMW_bias_sum = 0
    pressure_bias_sum = 0

    wind_RMSE_sum = 0
    RMW_RMSE_sum = 0
    pressure_RMSE_sum = 0

    item = 0

    wind_output_list = []
    wind_label_list = []
    RMW_output_list = []
    RMW_label_list = []
    wind_MAE_list_mine = []
    RMW_MAE_list_mine = []
    with torch.no_grad():
        for batch, data in enumerate(tqdm(dataset)):
            inputs = data["npy"]
            inputs = inputs.to(config.device)

            diff_inputs = data["diff_npy"]
            diff_inputs = diff_inputs.to(config.device)

            wind_label = data["wind"].to(config.device)
            RMW_label = data["RMW"].to(config.device)
            pressure_label = data["pressure"].to(config.device)

            pre = data['pre'].to(config.device)
            pre = pre.to(torch.float32)
            pre = pre.reshape(pre.shape[0], 1)

            t = data['occur_t'].to(config.device)
            t = t.to(torch.float32)
            t = t.reshape(t.shape[0], 1)

            lat = data['lat'].to(config.device)
            lat = lat.to(torch.float32)
            lat = lat.reshape(lat.shape[0], 1)

            lon = data['lon'].to(config.device)
            lon = lon.to(torch.float32)
            lon = lon.reshape(lon.shape[0], 1)

            wind, RMW = model(inputs, diff_inputs, pre, lat, lon, t)

            wind = wind[:, 0]
            RMW = RMW[:, 0]

            # 标签和估计结果反归一化后比较误差
            wind_label_re = wind_label.cpu().detach().numpy() * (170 - 19) + 19
            wind_re = wind.cpu().detach().numpy() * (170 - 19) + 19
            wind_error = wind_error + np.sum(np.abs(wind_re - wind_label_re))
            wind_bias_sum = wind_bias_sum + np.sum(wind_re - wind_label_re)
            wind_RMSE_sum = wind_RMSE_sum + np.sum((wind_re - wind_label_re) * (wind_re - wind_label_re))

            RMW_label_re = RMW_label.cpu().detach().numpy() * (200 - 5) + 5
            RMW_re = RMW.cpu().detach().numpy() * (200 - 5) + 5
            RMW_error = RMW_error + np.sum(np.abs(RMW_re - RMW_label_re))
            RMW_bias_sum = RMW_bias_sum + np.sum(RMW_re - RMW_label_re)
            RMW_RMSE_sum = RMW_RMSE_sum + np.sum((RMW_re - RMW_label_re) * (RMW_re - RMW_label_re))

            item += 1

            for i in range(0, len(wind_re)):
                wind_output_list.append(wind_re[i])
                wind_label_list.append(wind_label_re[i])
                RMW_output_list.append(RMW_re[i])
                RMW_label_list.append(RMW_label_re[i])
                wind_MAE_list_mine.append(np.abs(wind_re - wind_label_re)[i])
                RMW_MAE_list_mine.append(np.abs(RMW_re - RMW_label_re)[i])
                print("wind_label={}, 估计结果：intensity={}".format(wind_label_re[i], wind_re[i]))
                print("RMW_label={}, 估计结果：RMW={}".format(RMW_label_re[i], RMW_re[i]))
                
    return wind_error, RMW_error, \
           wind_bias_sum, wind_RMSE_sum, RMW_bias_sum, RMW_RMSE_sum, \
           wind_label_list, wind_output_list, RMW_label_list, RMW_output_list, \
           wind_MAE_list_mine, RMW_MAE_list_mine


if __name__ == '__main__':
    test_transform = None
    test_dataset = Dataset(config.predict_npy_path156, config.predict_npy_path_diff, test_transform, config.data_format)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
    )

    wind_error, RMW_error, \
    wind_bias_sum, wind_RMSE_sum, RMW_bias_sum, RMW_RMSE_sum, \
    wind_label_list, wind_output_list, RMW_label_list, RMW_output_list, \
    wind_MAE_list_mine, RMW_MAE_list_mine = test_npys(test_dataloader)

    test_num = len(wind_label_list)

    wind_error = wind_error / test_num
    RMW_error = RMW_error / test_num
    wind_bias = wind_bias_sum / test_num
    wind_RMSE = math.sqrt(wind_RMSE_sum / test_num)
    RMW_bias = RMW_bias_sum / test_num
    RMW_RMSE = math.sqrt(RMW_RMSE_sum / test_num)
    print("强度估计值的平均误差={}".format(wind_error))
    print("风圈RMW估计值的平均误差={}".format(RMW_error))
    print("强度估计的bias={}".format(wind_bias))
    print("风圈RMW估计的bias={}".format(RMW_bias))
    
    print("强度估计的平均RMSE={}".format(wind_RMSE))
    print("风圈RMW估计的RMSE={}".format(RMW_RMSE))

