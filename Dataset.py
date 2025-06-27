import numpy as np
import torch
import os
from tqdm import tqdm
import AConfig as config
from datetime import datetime, timedelta
import pickle

def get_previous_npy_index(npy_index, x):
    """
    根据给定的 npy_index 和小时数 x，返回前 x 小时的 npy_index。
    npy_index 格式: '年份_台风名字_时间'，例如 '2019_台风名字_2019010506'
    """
    # 解析 npy_index
    parts = npy_index.split('_')
    year_typhoon_name = parts[0] + '_' + parts[1]
    time_str = parts[2]  # 例如 '2019010506'

    # 将时间字符串转换为 datetime 对象
    time_format = '%Y%m%d%H'
    time_obj = datetime.strptime(time_str, time_format)

    # 减去 x 小时
    previous_time_obj = time_obj - timedelta(hours=x)

    # 将 datetime 对象转换回字符串
    previous_time_str = previous_time_obj.strftime(time_format)

    # 重新拼接生成新的 npy_index
    previous_npy_index = year_typhoon_name + "_" + previous_time_str

    return previous_npy_index

def default_loader(path):
    raw_data = np.load(path, allow_pickle=True)
    tensor_data = torch.from_numpy(raw_data)
    tensor_data = tensor_data.type(torch.FloatTensor)
    return tensor_data

def load_dict_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

'''full整体归一'''
def get_fnorm_now_chw(x, statistic_dic):
    x_norm = x.clone()
    for c in range(4):
        '''处理负值（按通道处理，用每个通道的均值填充）'''
        # channel_mean = torch.mean(x[c][x[c] > 0])
        # x_norm[c] = torch.where(x[c] < 0, 0, x[c])
        '''train'''
        # maxv = statistic_dic['now_train'][c][0]
        # minv = statistic_dic['now_train'][c][1]
        '''all'''
        maxv = statistic_dic['now_all'][c][0]
        minv = statistic_dic['now_all'][c][1]
        x_norm[c] = (x[c] - minv) / (maxv - minv)

    return x_norm

def get_fnorm_p3h_chw(x, statistic_dic):
    x_norm = x.clone()
    for c in range(4):
        '''处理负值（按通道处理，用每个通道的均值填充）'''
        # channel_mean = torch.mean(x[c][x[c] > 0])
        # x_norm[c] = torch.where(x[c] <= 0, 0, x[c])
        '''train'''
        # maxv = statistic_dic['p3h_train'][c][0]
        # minv = statistic_dic['p3h_train'][c][1]
        '''all'''
        maxv = statistic_dic['p3h_all'][c][0]
        minv = statistic_dic['p3h_all'][c][1]
        x_norm[c] = (x[c] - minv) / (maxv - minv)
    return x_norm

class Findpxh_Dataset_k_pr():
    def __init__(self, k8_path, pxh, data_transforms=None, data_format='npy'):

        self.data_transforms = data_transforms
        self.data_format = data_format
        self.k8_paths = k8_path
        self.pxh = pxh
        self.k8_btemps = []
        self.pxh_k8_btemps = []

        self.msws = []
        self.rmws = []
        self.r34s = []
        self.mslps = []

        self.lats = []
        self.lons = []
        self.ts = []
        self.levels = []
        self.plevels = []

        self.pre_tcfs = []
        self.pre_isrs = []
        self.pre_pwrs = []
        self.pre_prrs = []
        self.pre_levels = []
        self.norm_ts = []
        # cc info
        self.cc100_dis_maxs, self.cc100_bt_mins, self.cc135_tb_maxs, \
        self.cc135_tb_means, self.cc135_tbds, self.cc135_nums, self.cc135_is = [], [], [], [], [], [], []
        # ic info
        self.ewtcs, self.ewgs, self.bt_ch7_stds, \
        self.bt_ch7_mins, self.bt_ch8_means, self.bt_ch13_mins = [], [], [], [], [], []
        # of info
        self.ocbts, self.trss, self.czts = [], [], []
        '''
        get BST Labels:
        TCName_Isotime --- [iso_time, lat, lon, t, level, mslp, msw, rmw, r34, sai_alpha]
        '''
        self.labels_dic = load_dict_from_pickle(config.labels_path)

        # 获取目录下的所有文件
        k8_files = os.listdir(k8_path)
        k8_files_set = set(k8_files)

        self.k8_sta_dic = load_dict_from_pickle(config.k8_sta_path)
        pre12h_labels_data = load_dict_from_pickle(config.p12hpr_pth)

        cc_dic_data = load_dict_from_pickle(config.cc_pth)
        ic_dic_data = load_dict_from_pickle(config.ic_pth)
        of_dic_data = load_dict_from_pickle(config.of_pth)

        # 遍历文件
        for i, filename in enumerate(k8_files):
            fname_split = filename.split("_")
            tcname = fname_split[1]
            year = fname_split[0]
            '''e.g. 2023_BOLAVEN_2023100809'''
            if len(fname_split) == 3:
                isotime = fname_split[2][:-4]
            else:
                isotime = fname_split[2]
            labels_dic_key = year + "_" + tcname + "_" + isotime

            '''e.g. BOLAVEN_2023100806'''
            p3h_labels_dic_key = get_previous_npy_index(labels_dic_key, 3)
            if len(fname_split) == 3:
                pxh_k8_fname = p3h_labels_dic_key + ".npy"
            else:
                if fname_split[3] == 'rotate315.npy':
                    continue
                pxh_k8_fname = p3h_labels_dic_key + "_" + fname_split[3]
            if filename in self.k8_sta_dic['now_inval_record'] or pxh_k8_fname in self.k8_sta_dic['p3h_inval_record']:
                continue
            if pxh_k8_fname not in k8_files_set:
                continue
            if p3h_labels_dic_key not in self.labels_dic.keys():
                continue
            if labels_dic_key not in self.labels_dic.keys():
                continue
            dic_index = str(isotime[:4]) + "_" + tcname + "_" + str(isotime)  # 例如2019_台风名字_2019010106
            if dic_index not in pre12h_labels_data.keys():
                continue

            lat, lon, t, level, mslp, msw, rmw, r34 = self.labels_dic[labels_dic_key][1:]
            pre_tcf, pre_isr, pre_pwr, pre_prr, pre_level, pre_mslp, pre_msw, pre_rmw, pre_r34 = pre12h_labels_data[
                dic_index]

            if dic_index not in cc_dic_data.keys():
                continue
            cc100_dis_max, cc100_bt_min, cc135_tb_max, cc135_tb_mean, cc135_tbd, cc135_num, cc135_i = cc_dic_data[dic_index]
            if dic_index not in ic_dic_data.keys():
                continue
            ewtc, ewg, bt_ch7_std, bt_ch7_min, bt_ch8_mean, bt_ch13_min = ic_dic_data[dic_index]
            ocbt, trs, czt = of_dic_data[dic_index]
            # cc info
            self.cc100_dis_maxs.append(cc100_dis_max)
            self.cc100_bt_mins.append(cc100_bt_min)
            self.cc135_tb_maxs.append(cc135_tb_max)
            self.cc135_tb_means.append(cc135_tb_mean)
            self.cc135_tbds.append(cc135_tbd)
            self.cc135_nums.append(cc135_num)
            self.cc135_is.append(cc135_i)
            # ic info
            self.ewtcs.append(ewtc)
            self.ewgs.append(ewg)
            self.bt_ch7_stds.append(bt_ch7_std)
            self.bt_ch7_mins.append(bt_ch7_min)
            self.bt_ch8_means.append(bt_ch8_mean)
            self.bt_ch13_mins.append(bt_ch13_min)
            # of info
            self.ocbts.append(ocbt)
            self.trss.append(trs)
            self.czts.append(czt)

            self.pre_tcfs.append(pre_tcf)
            self.pre_isrs.append(pre_isr)
            self.pre_pwrs.append(pre_pwr)
            self.pre_prrs.append(pre_prr)
            # self.plevels.append(int(pre_level))
            self.plevels.append(pre_level)

            self.pxh_k8_btemps.append(k8_path + pxh_k8_fname)
            self.k8_btemps.append(k8_path + filename)
            self.msws.append(msw)
            self.rmws.append(rmw)
            self.r34s.append(r34)
            self.mslps.append(mslp)

            self.lats.append(lat)
            self.lons.append(lon)
            self.ts.append(t)
            self.levels.append(level)
            self.norm_ts.append(t)

        print("msws max = {}, min = {}".format(max(self.msws), min(self.msws)))
        print("rmws max = {}, min = {}".format(max(self.rmws), min(self.rmws)))
        print("r34s max = {}, min = {}".format(max(self.r34s), min(self.r34s)))
        print("mslps max = {}, min = {}".format(max(self.mslps), min(self.mslps)))
        print("lat max = {}, min = {}".format(max(self.lats), min(self.lats)))
        print("lon max = {}, min = {}".format(max(self.lons), min(self.lons)))
        print("t max = {}, min = {}".format(max(self.ts), min(self.ts)))
        print("pre_tcf max = {}, min = {}".format(max(self.pre_tcfs), min(self.pre_tcfs)))
        print("pre_isr max = {}, min = {}".format(max(self.pre_isrs), min(self.pre_isrs)))
        print("pre_pwr max = {}, min = {}".format(max(self.pre_pwrs), min(self.pre_pwrs)))
        print("pre_prr max = {}, min = {}".format(max(self.pre_prrs), min(self.pre_prrs)))

        print("cc100_dis_max max = {}, min = {}".format(max(self.cc100_dis_maxs), min(self.cc100_dis_maxs)))
        print("cc100_bt_min max = {}, min = {}".format(max(self.cc100_bt_mins), min(self.cc100_bt_mins)))
        print("cc135_tb_max max = {}, min = {}".format(max(self.cc135_tb_maxs), min(self.cc135_tb_maxs)))
        print("cc135_tb_mean max = {}, min = {}".format(max(self.cc135_tb_means), min(self.cc135_tb_means)))
        print("cc135_tbd max = {}, min = {}".format(max(self.cc135_tbds), min(self.cc135_tbds)))
        print("cc135_num max = {}, min = {}".format(max(self.cc135_nums), min(self.cc135_nums)))
        print("cc135_i max = {}, min = {}".format(max(self.cc135_is), min(self.cc135_is)))

        print("ewtc max = {}, min = {}".format(max(self.ewtcs), min(self.ewtcs)))
        print("ewg max = {}, min = {}".format(max(self.ewgs), min(self.ewgs)))
        print("bt_ch7_std max = {}, min = {}".format(max(self.bt_ch7_stds), min(self.bt_ch7_stds)))
        print("bt_ch7_min max = {}, min = {}".format(max(self.bt_ch7_mins), min(self.bt_ch7_mins)))
        print("bt_ch8_mean max = {}, min = {}".format(max(self.bt_ch8_means), min(self.bt_ch8_means)))
        print("bt_ch13_min max = {}, min = {}".format(max(self.bt_ch13_mins), min(self.bt_ch13_mins)))

        print("ocbt max = {}, min = {}".format(max(self.ocbts), min(self.ocbts)))
        print("trs max = {}, min = {}".format(max(self.trss), min(self.trss)))
        print("czt max = {}, min = {}".format(max(self.czts), min(self.czts)))

        # 标签归一化
        for i in range(len(self.msws)):
            '''pre 12 h PR'''
            self.msws[i] = (self.msws[i] - 35) / (170 - 35)
            self.rmws[i] = (self.rmws[i] - 5) / (130 - 5)
            self.lats[i] = (self.lats[i] - (-32.038)) / (42.491 - (-32.038))
            self.lons[i] = (self.lons[i] - 83.892) / (195.765 - 83.892)
            self.norm_ts[i] = (self.ts[i] - 12) / (459 - 12)
            self.pre_tcfs[i] = (self.pre_tcfs[i] - (-2)) / (0.98 - (-2))
            self.pre_isrs[i] = (self.pre_isrs[i] - 0.22) / (34 - 0.22)
            self.pre_pwrs[i] = (self.pre_pwrs[i] - (5.18)) / (28.77 - (5.18))
            self.pre_prrs[i] = (self.pre_prrs[i] - (2.76)) / (99.9 - (2.76))
            # cc
            self.cc100_dis_maxs[i] = (self.cc100_dis_maxs[i] - 28.28) / (100 - 28.28)
            self.cc100_bt_mins[i] = (self.cc100_bt_mins[i] - 171.98) / (242.66 - 171.98)
            self.cc135_tb_maxs[i] = (self.cc135_tb_maxs[i] - 187.56) / (244.15 - 187.56)
            self.cc135_tb_means[i] = (self.cc135_tb_means[i] - 181.92) / (240.24 - 181.92)
            self.cc135_tbds[i] = (self.cc135_tbds[i] - 0) / (67.65 - 0)
            self.cc135_nums[i] = (self.cc135_nums[i] - 1) / (395 - 1)
            self.cc135_is[i] = (self.cc135_is[i] - 185.38) / (240.88 - 185.38)
            # ic
            self.ewtcs[i] = (self.ewtcs[i] - (-73.27)) / (36.09 - (-73.27))
            self.ewgs[i] = (self.ewgs[i] - (-12.21)) / (6.02 - (-12.21))
            self.bt_ch7_stds[i] = (self.bt_ch7_stds[i] - 1.49) / (30.19 - 1.49)
            self.bt_ch7_mins[i] = (self.bt_ch7_mins[i] - 174.77) / (231.2 - 174.77)
            self.bt_ch8_means[i] = (self.bt_ch8_means[i] - 186.28) / (265.18 - 186.28)
            self.bt_ch13_mins[i] = (self.bt_ch13_mins[i] - 171.98) / (239.05 - 171.98)
            # of
            self.ocbts[i] = (self.ocbts[i] - 192.68) / (282.79 - 192.68)
            self.trss[i] = (self.trss[i] - 184.72) / (283.92 - 184.72)
            self.czts[i] = (self.czts[i] - 192.68) / (282.79 - 192.68)

    def __len__(self):
        return len(self.k8_btemps)

    def __getitem__(self, index):
        # 4, 156, 156
        btemp_file_path = self.k8_btemps[index]
        k8_btemp = default_loader(btemp_file_path)
        k8_btemp = get_fnorm_now_chw(k8_btemp, self.k8_sta_dic)

        # 4, 156, 156
        pxh_btemp_file_path = self.pxh_k8_btemps[index]
        pxh_k8_btemp = default_loader(pxh_btemp_file_path)
        pxh_k8_btemp = get_fnorm_p3h_chw(pxh_k8_btemp, self.k8_sta_dic)

        '''diff compute'''
        ch7_btd_3h = k8_btemp[0] - pxh_k8_btemp[0]
        ch13_btd_3h = k8_btemp[2] - pxh_k8_btemp[2]
        btd_ch8_13 = k8_btemp[1] - k8_btemp[2]
        btd_ch13_15 = k8_btemp[2] - k8_btemp[3]
        btd_list = [i.unsqueeze(dim=0) for i in [ch7_btd_3h, ch13_btd_3h, btd_ch8_13, btd_ch13_15]]
        btdiff = torch.cat(btd_list, dim=0)[:, 78 - 50: 78 + 50, 78 - 50: 78 + 50]

        cc100_dis_max = self.cc100_dis_maxs[index]
        cc100_bt_min = self.cc100_bt_mins[index]
        cc135_tb_max = self.cc135_tb_maxs[index]
        cc135_tb_mean = self.cc135_tb_means[index]
        cc135_tbd = self.cc135_tbds[index]
        cc135_num = self.cc135_nums[index]
        cc135_i = self.cc135_is[index]
        # ic info
        ewtc = self.ewtcs[index]
        ewg = self.ewgs[index]
        bt_ch7_std = self.bt_ch7_stds[index]
        bt_ch7_min = self.bt_ch7_mins[index]
        bt_ch8_mean = self.bt_ch8_means[index]
        bt_ch13_min = self.bt_ch13_mins[index]
        # of info
        ocbt = self.ocbts[index]
        trs = self.trss[index]
        czt = self.czts[index]

        msw = self.msws[index]
        rmw = self.rmws[index]
        r34 = self.r34s[index]
        mslp = self.mslps[index]

        lat = self.lats[index]
        lon = self.lons[index]
        level = self.levels[index]
        t = self.ts[index]
        norm_t = self.norm_ts[index]
        pre_tcf = round(self.pre_tcfs[index], 3)
        pre_isr = round(self.pre_isrs[index], 3)
        pre_pwr = round(self.pre_pwrs[index], 3)
        pre_prr = round(self.pre_prrs[index], 3)
        pre_level = self.plevels[index]

        # ccinfo = torch.tensor([cc100_dis_max, cc100_bt_min, cc135_tb_max, cc135_tb_mean, cc135_tbd, cc135_num, cc135_i])
        ccinfo = torch.tensor([cc135_tb_max, cc135_num, cc135_i])
        dev_sh = torch.tensor([bt_ch7_min, bt_ch8_mean, bt_ch13_min])
        dev_spv = torch.tensor([ewtc, ewg, bt_ch7_std])
        dev_spr = torch.tensor([ocbt, trs, czt])
        dev_level = torch.tensor([pre_level - 1, pre_level, pre_level + 1])

        sample = {'lat': lat, 'lon': lon, 'occur_t': t, 'cat': level, 'norm_t': norm_t,
                  'pre_level': pre_level, 'r34': r34, 'rmw': rmw, 'msw': msw, 'mslp': mslp,
                  'pre_tcf': pre_tcf, 'pre_isr': pre_isr, 'pre_pwr': pre_pwr, 'pre_prr': pre_prr,
                  'k8_btemp': k8_btemp,
                  'pxh_k8_btemp': pxh_k8_btemp, 'btdiff': btdiff,
                  # 'cc100_dis_max': cc100_dis_max, 'cc100_bt_min': cc100_bt_min, 'cc135_tb_max': cc135_tb_max,
                  # 'cc135_tb_mean': cc135_tb_mean, 'cc135_tbd': cc135_tbd, 'cc135_num': cc135_num, 'cc135_i': cc135_i,
                  'ewtc': ewtc, 'ewg': ewg, 'bt_ch7_std': bt_ch7_std, 'bt_ch7_min': bt_ch7_min,
                  'bt_ch8_mean': bt_ch8_mean, 'bt_ch13_min': bt_ch13_min,
                  'ocbt': ocbt, 'trs': trs, 'czt': czt,
                  'ccinfo': ccinfo, 'ccnum': cc135_num, 'moi': bt_ch8_mean,
                  'dev_sh': dev_sh, 'dev_spv': dev_spv, 'dev_spr': dev_spr, 'dev_level': dev_level
                  }
        return sample

if __name__ == '__main__':
    '''find pxh npy in single t npy'''
    print("trainset label max/min v:")
    # train_dataset = Findpxh_Dataset_k('/opt/data/private/norm_data_npy/Full2015_2023_Dataets/k89_4ch1/train/', 3, None, config.data_format)
    train_dataset = Findpxh_Dataset_k_pr(config.train_k8_path, 3, None, config.data_format)
    print("validset label max/min v:")
    valid_dataset = Findpxh_Dataset_k_pr(config.valid_k8_path, 3, None, config.data_format)
    print("testset label max/min v:")
    test_dataset = Findpxh_Dataset_k_pr(config.predict_k8_path, 3, None, config.data_format)
    # for batch, data in enumerate(tqdm(test_dataset)):
    #     btdiff = data["btdiff"]
    #     print(btdiff.size())