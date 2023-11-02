import glob
import numpy as np
import torch
import os
from openpyxl.reader.excel import load_workbook
from tqdm import tqdm
import Config as config

def default_loader(path):
    raw_data = np.load(path, allow_pickle=True)
    tensor_data = torch.from_numpy(raw_data)

    return tensor_data

class Dataset():
    def __init__(self, data_root_path_156, data_root_path_diff, data_transforms=None, data_format='npy'):

        self.data_transforms = data_transforms
        self.data_format = data_format
        self.npy_156_paths = data_root_path_156
        self.npy_diff_paths = data_root_path_diff

        self.npys_156 = []
        self.npys_diff = []
        self.winds = []
        self.RMWs = []
        self.R34s = []
        self.pressures = []
        self.pres = []
        self.lats = []
        self.lons = []
        self.ts = []

        for npy_file in os.listdir(data_root_path_156):
            filename = npy_file.split("_")
            lat = float(filename[0])
            lon = float(filename[1])
            t = float(filename[2])
            pre = float(filename[3])
            pressure = float(filename[4])
            wind = float(filename[5])
            RMW = float(filename[6])
            R34 = float(filename[7])

            self.npys_156.append(data_root_path_156 + npy_file)
            self.npys_diff.append(data_root_path_diff + npy_file)

            self.winds.append(wind)
            self.RMWs.append(RMW)
            self.R34s.append(R34)
            self.pres.append(pre)
            self.pressures.append(pressure)
            self.lats.append(lat)
            self.lons.append(lon)
            self.ts.append(t)

        wind_min = min(self.winds)
        wind_max = max(self.winds)
        RMW_min = min(self.RMWs)
        RMW_max = max(self.RMWs)
        # R34_min = min(self.R34s)
        # R34_max = max(self.R34s)
        # pressure_min = min(self.pressures)
        # pressure_max = max(self.pressures)

        print("wind max = {}".format(wind_max))
        print("wind min = {}".format(wind_min))
        print("RMW max = {}".format(RMW_max))
        print("RMW min = {}".format(RMW_min))
        # print("R34 max = {}".format(R34_max))
        # print("R34 min = {}".format(R34_min))
        # print("pressure max = {}".format(pressure_max))
        # print("pressure min = {}".format(pressure_min))

        # Label normalization
        for i in range(len(self.winds)):
            self.winds[i] = (self.winds[i] - 19) / (170 - 19)
            self.RMWs[i] = (self.RMWs[i] - 5) / (200 - 5)
            # self.R34s[i] = (self.R34s[i] - 11.25) / (406.25 - 11.25)
            self.pressures[i] = (self.pressures[i] - 882) / (1013 - 882)
            self.lats[i] = (self.lats[i] - (-32.0377)) / (43.772 - (-32.0377))
            self.lons[i] = (self.lons[i] - 86.25) / (193.7 - 86.25)
            # self.ts[i] = (self.ts[i] - 459) / (459 - 3)


    def __len__(self):
        return len(self.npys_156)

    def __getitem__(self, index):

        npy_file_path_156 = self.npys_156[index]
        npy_156 = default_loader(npy_file_path_156)

        npy_file_path_diff = self.npys_diff[index]
        npy_diff = default_loader(npy_file_path_diff)

        if self.data_transforms is not None:
            npy_156 = self.data_transforms(npy_156)
            npy_diff = self.data_transforms(npy_diff)

        wind = self.winds[index]
        RMW = self.RMWs[index]
        R34 = self.R34s[index]
        pressure = self.pressures[index]
        pre = self.pres[index]
        lat = self.lats[index]
        lon = self.lons[index]
        t = self.ts[index]

        sample = {'lat': lat, 'lon': lon, 'occur_t': t, 'pre': pre, 'npy': npy_156, 'diff_npy': npy_diff,
                  'RMW': RMW, 'wind': wind, 'pressure': pressure}

        return sample
