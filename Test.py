import math
import time
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score
import numpy as np
from matplotlib import image
from tqdm import tqdm
from Dataset import *
import Config as config
import Set_seed
transform_test = None

def test_model(model, dataset):

    model.eval()

    item = 0

    batch_num = 0

    wind_error = 0
    wind_RMSE_sum = 0
    wind_bias_sum = 0

    rmw_error = 0
    rmw_RMSE_sum = 0
    rmw_bias_sum = 0

    wind_output_list = []
    wind_label_list = []
    RMW_output_list = []
    RMW_label_list = []
    v_errors = []
    r1_errors = []
    cat_vbias = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    cat_rbias = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    num = 0
    with torch.no_grad():
        for batch, data in enumerate(tqdm(dataset)):
            k8_btemp = data["k8_btemp"]
            k8_btemp = k8_btemp.to(config.device)

            pxh_k8_btemp = data["pxh_k8_btemp"]
            pxh_k8_btemp = pxh_k8_btemp.to(config.device)
            '''cat pxh and now'''
            btemp = torch.cat([pxh_k8_btemp, k8_btemp], dim=1)

            btdiff = data["btdiff"]
            btdiff = btdiff.to(config.device)

            t = data['norm_t'].to(torch.float32).to(config.device).unsqueeze(dim=-1)

            msw_label = data["msw"]
            msw_label = msw_label.to(config.device)
            rmw_label = data["rmw"]
            rmw_label = rmw_label.to(config.device)

            plevel = data["pre_level"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            cat = data["cat"].to(config.device)
            lat = data["lat"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            lon = data["lon"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            ccinfo = data["ccinfo"].to(torch.float32).to(config.device)  # b, 7

            pre_tcf = data["pre_tcf"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_isr = data["pre_isr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_pwr = data["pre_pwr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_prr = data["pre_prr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_pr = torch.cat([pre_tcf, pre_isr, pre_pwr, pre_prr], dim=-1)

            ccnum = data["ccnum"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            moi = data["moi"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            ic = data["bt_ch7_std"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            of = data["ocbt"].to(torch.float32).to(config.device).unsqueeze(dim=-1)

            b = cat.size(0)

            msw, rmw = model(btemp, btdiff, moi, ic, of, plevel, lat, lon, t, ccnum, pre_tcf, pre_pwr)

            batch_num += 1
            num += b
            print("wind_label={}".format(msw_label))
            print("Intensity={}".format(msw))
            print("rmw_label={}".format(rmw_label))
            print("RMW={}".format(rmw))

            # 反归一化
            wind_label_re = msw_label.cpu().detach().numpy() * (170 - 35) + 35
            wind_re = msw.cpu().detach().numpy() * (170 - 35) + 35
            wind_error = wind_error + np.sum(np.abs(wind_re - wind_label_re))
            wind_RMSE_sum = wind_RMSE_sum + np.sum((wind_re - wind_label_re) * (wind_re - wind_label_re))
            wind_bias_sum = wind_bias_sum + np.sum(wind_re - wind_label_re)

            rmw_label_re = rmw_label.cpu().detach().numpy() * (130 - 5) + 5
            rmw_re = rmw.cpu().detach().numpy() * (130 - 5) + 5
            rmw_error = rmw_error + np.sum(np.abs(rmw_re - rmw_label_re))
            rmw_RMSE_sum = rmw_RMSE_sum + np.sum((rmw_re - rmw_label_re) * (rmw_re - rmw_label_re))
            rmw_bias_sum = rmw_bias_sum + np.sum(rmw_re - rmw_label_re)

            item += rmw_label.shape[0]

            for i in range(0, len(rmw_re)):
                wind_output_list.append(wind_re[i])
                wind_label_list.append(wind_label_re[i])
                RMW_output_list.append(rmw_re[i])
                RMW_label_list.append(rmw_label_re[i])
                v_errors.append(np.abs(wind_re[i] - wind_label_re[i]))
                r1_errors.append(np.abs(rmw_re[i] - rmw_label_re[i]))
                cat_vbias[int(cat[i])].append(np.abs(wind_re[i] - wind_label_re[i]))
                cat_rbias[int(cat[i])].append(np.abs(rmw_re[i] - rmw_label_re[i]))
    v_err_mean, v_err_var = np.mean(v_errors), np.std(v_errors)
    r1_err_mean, r1_err_var = np.mean(r1_errors), np.std(r1_errors)
    vmape = np.mean(np.abs((np.array(wind_output_list) - np.array(wind_label_list)) / np.array(wind_label_list))) * 100
    rmape = np.mean(np.abs((np.array(RMW_output_list) - np.array(RMW_label_list)) / np.array(RMW_label_list))) * 100
    vsmape = np.mean(np.abs(np.array(wind_output_list) - np.array(wind_label_list)) / (
                np.abs(np.array(wind_output_list)) + np.abs(np.array(wind_label_list))) / 2) * 100
    rsmape = np.mean(np.abs(np.array(RMW_output_list) - np.array(RMW_label_list)) / (
                np.abs(np.array(RMW_output_list)) + np.abs(np.array(RMW_label_list))) / 2) * 100
    print("estimation v mape = {}, smape = {}".format(vmape, vsmape))
    print("estimation r mape = {}, smape = {}".format(rmape, rsmape))
    return wind_error, wind_RMSE_sum, wind_bias_sum, wind_label_list, wind_output_list, \
           rmw_error, rmw_RMSE_sum, rmw_bias_sum, RMW_label_list, RMW_output_list

if __name__ == '__main__':
    Set_seed.set_seed(1)
    model = torch.load(config.predict_model, map_location='cuda:0').to(config.device)
    test_transform = None
    test_dataset = Findpxh_Dataset_k_pr(config.predict_k8_path, 3, None, config.data_format)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )

    ''' statistic  '''
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    torch.cuda.synchronize()
    start = time.time()

    wind_error, wind_RMSE_sum, wind_bias_sum, wind_label_list, wind_output_list, \
    rmw_error, rmw_RMSE_sum, rmw_bias_sum, RMW_label_list, RMW_output_list = test_model(model, test_dataloader)

    torch.cuda.synchronize()
    end = time.time()
    print('infer_time:', end - start)

    test_num = len(RMW_label_list)

    test_msw_error = wind_error / test_num
    msw_RMSE = math.sqrt(wind_RMSE_sum / test_num)
    print(f'Mean MSW MAE: {test_msw_error:.2f}')
    print(f'Mean MSW RMSE: {msw_RMSE:.2f}')

    test_rmw_error = rmw_error / test_num
    rmw_RMSE = math.sqrt(rmw_RMSE_sum / test_num)
    print(f'Mean RMW MAE: {test_rmw_error:.2f}')
    print(f'Mean RMW RMSE: {rmw_RMSE:.2f}')
    msw_bias = np.mean(np.array(wind_label_list) - np.array(wind_output_list))
    print(f' msw bias: {msw_bias:.2f}')
    rmw_bias = np.mean(np.array(RMW_label_list) - np.array(RMW_output_list))
    print(f' rmw bias: {rmw_bias:.2f}')

