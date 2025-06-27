from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import AConfig as config

class Shared_Network156_chres(nn.Module):
    def __init__(self):
        super().__init__()
        self.Lat_FC = nn.Sequential(OrderedDict(
            [('Lat_FC1', nn.Linear(1, 5)),
             ('Tanh', nn.Tanh()),
             ('Lat_FC2', nn.Linear(5, 25)),
             ('Tanh', nn.Tanh()),
             ('Lat_FC3', nn.Linear(25, 58)),
             ('Tanh', nn.Tanh()),
             ('Lat_FC4', nn.Linear(58, 156)),
             ('Tanh', nn.Tanh())]))
        self.Lon_FC = nn.Sequential(OrderedDict(
            [('Lon_FC1', nn.Linear(1, 5)),
             ('Tanh', nn.Tanh()),
             ('Lon_FC2', nn.Linear(5, 25)),
             ('Tanh', nn.Tanh()),
             ('Lon_FC3', nn.Linear(25, 58)),
             ('Tanh', nn.Tanh()),
             ('Lon_FC4', nn.Linear(58, 156)),
             ('Tanh', nn.Tanh())]))
        self.conv1 = nn.Conv2d(9, 16, kernel_size=7, stride=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x, lat, lon):
        # bs, 156
        lat = self.Lat_FC(lat)
        # print(lat)
        # bs, 156, 1
        lat = lat.unsqueeze(dim=2)
        # bs, 156
        lon = self.Lon_FC(lon)
        # print(lon)
        # bs, 1, 156
        lon = lon.unsqueeze(dim=1)
        # bs, 156, 156
        geo_aux = torch.bmm(lat, lon)
        # bs, 1, 156, 156
        geo_aux = geo_aux.unsqueeze(dim=1)
        # bs, 8, 156, 156
        # x = x * geo_aux
        # bs, 9, 156, 156
        x = torch.cat((x, geo_aux), dim=1)

        x = self.conv1(x)  # (16, 50, 50)
        x = self.relu(x)
        x = self.pool1(x)  # (16, 24, 24)

        x = self.conv2(x)  # (32, 20, 20)
        x = self.relu(x)
        x = self.pool2(x)  # (32, 9, 9)
        res_x1 = x

        x = self.conv3(x)  # (64, 9, 9)
        x = self.relu(x)
        x = torch.cat((res_x1, x), dim=1)       # (96, 9, 9)
        x = self.pool3(x)  # (96, 4, 4)
        res_x2 = x

        x = self.conv4(x)  # (192, 4, 4)
        x = self.relu(x)
        x = torch.cat((res_x2, x), dim=1)       # (96 + 192=288, 4, 4)
        x = self.pool4(x)  # (288, 2, 2)
        # x = self.flatten(x)
        return x

class GraphEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, device):
        super(GraphEncoder, self).__init__()

        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)
        self.device = device

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    def reparametrize(self, mu, logstd):
        return mu + torch.randn_like(logstd) * torch.exp(logstd)

    def forward(self, data):
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        mu, logvar = self.encode(x, edge_index)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

class GraphDecoder(nn.Module):
    def __init__(self, out_channels, in_channels):
        super(GraphDecoder, self).__init__()
        self.MLP_spv = nn.Sequential(
            nn.Linear(out_channels, 24),
            nn.ReLU(),
            nn.Linear(24, in_channels)
        )
        self.MLP_spr = nn.Sequential(
            nn.Linear(out_channels, 24),
            nn.ReLU(),
            nn.Linear(24, in_channels)
        )
    def forward(self, z):
        adj = torch.bmm(z, z.transpose(1, 2))
        x_spv = self.MLP_spv(z)
        x_spr = self.MLP_spr(z)
        return x_spv, x_spr, torch.sigmoid(adj)

def GraphD_Construt(nodef, adj):
    # 构造边索引（Edge Index）和边权重（Edge Weight）
    edge_index = []
    edge_weight = []
    b, n, d = nodef.size()
    # 构造输入和估计值之间的边
    for i in range(n):
        for j in range(n):
            if adj[i, j] != 0:
                edge_index.append([i, j])  # 从输入节点到输出节点
                edge_index.append([j, i])  # 从输出节点到输入节点
                edge_weight.append(adj[i, j])
                edge_weight.append(adj[i, j])

    edge_index = torch.tensor(edge_index).t().contiguous()  # 转置并转换为tensor
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    data = Data(x=nodef, edge_index=edge_index, edge_attr=edge_weight)
    return data

class Node_Enc(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.MLP_sh = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, out_dim)
        )
        self.MLP_spv = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, out_dim)
        )
        self.MLP_spr = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, out_dim)
        )
        self.MLP_level = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, out_dim)
        )
    def forward(self, dev_sh, dev_spv, dev_spr, dev_level):
        # b, out_dim
        dev_shf = self.MLP_sh(dev_sh)
        dev_spvf = self.MLP_sh(dev_spv)
        dev_sprf = self.MLP_sh(dev_spr)
        dev_levelf = self.MLP_sh(dev_level)
        # b, 1, out_dim in list
        node_list = [i.unsqueeze(dim=1) for i in [dev_spvf, dev_shf, dev_levelf, dev_sprf]]
        nodef = torch.cat(node_list, dim=1)  # b, node_num, out_dim
        return nodef

class Cause2DevGuid(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_enc = Node_Enc(in_dim=3, out_dim=16)
        self.cause_enc = GraphEncoder(in_channels=16, out_channels=16, device=config.device)
        self.cause_dec = GraphDecoder(out_channels=16, in_channels=16)

    def forward(self, dev_sh, dev_spv, dev_spr, dev_level):
        adj = torch.eye(4)
        nodef = self.node_enc(dev_sh, dev_spv, dev_spr, dev_level)  # b, node_num=4, out_dim=16
        gdata = GraphD_Construt(nodef, adj)
        # b, node_num=4, out_dim
        z, mu, logvar = self.cause_enc(gdata)
        # b, node_num=4, out_dim
        x_spv, x_spr, adj_dec = self.cause_dec(z)
        return x_spv, x_spr, adj_dec

class Dev_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 4, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=0)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        self.conv2 = nn.Conv3d(4, 8, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=0)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        self.conv3 = nn.Conv3d(8, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        self.conv4 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        self.conv5 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        self.cause = Cause2DevGuid()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, diff_x, dev_sh, dev_spv, dev_spr, dev_level):
        # bs, 1, 4, 58, 58
        diff_x = diff_x.unsqueeze(dim=1)

        diff_x = self.conv1(diff_x)  # 4, 4, 98, 98
        diff_x = self.relu(diff_x)
        diff_x = self.pool1(diff_x)  # 4, 4, 49, 49

        diff_x = self.conv2(diff_x)  # 8, 4, 47, 47
        diff_x = self.relu(diff_x)
        diff_x = self.pool2(diff_x)  # 8, 4, 23, 23

        diff_x = self.conv3(diff_x)  # 16, 4, 21, 21
        diff_x = self.relu(diff_x)
        diff_x = self.pool3(diff_x)  # 16, 4, 10, 10

        diff_x = self.conv4(diff_x)  # 16, 4, 8, 8
        diff_x = self.relu(diff_x)
        diff_x = self.pool4(diff_x)  # 16, 4, 4, 4

        diff_x = self.conv5(diff_x)  # 16, 4, 4, 4
        diff_x = self.relu(diff_x)
        diff_x = self.pool5(diff_x)  # 16, 4, 2, 2
        b = diff_x.size(0)
        diff_x = diff_x.view(b, 16, 4, -1).transpose(1,3)
        # b, 4, 16
        x_spv, x_spr, adj_dec = self.cause(dev_sh, dev_spv, dev_spr, dev_level)
        # b, 1, 4, 16 * b, 4, 4, 16
        dev_spvf = x_spv.unsqueeze(dim=1) * diff_x
        dev_sprf = x_spr.unsqueeze(dim=1) * diff_x

        dev_spvf = self.flatten(dev_spvf)
        dev_sprf = self.flatten(dev_sprf)

        return dev_spvf, dev_sprf, adj_dec

class CCTCFPWRsh_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Shared_Network156_chres()

        self.cc_enc = nn.Sequential(
            nn.Linear(1, 4)
        )
        self.tcf_enc = nn.Sequential(
            nn.Linear(1, 4)
        )
        self.pwr_enc = nn.Sequential(
            nn.Linear(1, 4)
        )
        self.t_enc = nn.Sequential(
            nn.Linear(1, 4)
        )
        self.gcn = GCNConv(4, 4)
        self.gcn_re = GCNConv(4, 4)

        self.flatten = nn.Flatten()

    def forward(self, x, lat, lon, t, cc, tcf, pwr):
        # 128, 2, 2 (hard share)
        shf = self.enc(x, lat, lon)     # b, 288, 2, 2
        b, shc, h, w = shf.size()
        shf = shf.view(b, shc, -1)    # b, 288, 4

        ccf = self.cc_enc(cc.unsqueeze(dim=-1))     # b, 7, 4
        tcff = self.tcf_enc(tcf.unsqueeze(dim=-1))  # b, 1, 4
        pwrf = self.pwr_enc(pwr.unsqueeze(dim=-1))  # b, 1, 4
        tf = self.t_enc(t.unsqueeze(dim=-1))  # b, 1, 4

        gf = torch.cat([ccf, tcff, pwrf, tf], dim=1)          # b, 10, 4

        adj = torch.eye(6)
        gf_gdata = GraphD_Construt(gf, adj)
        gf_x, gf_edge_index = gf_gdata.x.to(config.device), gf_gdata.edge_index.to(config.device)
        gf = self.gcn(gf_x, gf_edge_index)  # b, 10, 4

        # gf_adj = torch.sigmoid(torch.bmm(gf, gf.transpose(1, 2)))       # b, 10, 10
        gf_adj = torch.mean(torch.sigmoid(torch.bmm(gf, gf.transpose(1, 2))), dim=0)  # b, 10, 10--->10, 10
        print("gf_adj: ", gf_adj)
        # gf_binadj = (gf_adj > torch.mean(gf_adj)).float()
        gf_binadj = (gf_adj > torch.mean(gf_adj, dim=0)).float()
        print("gf_binadj: ", gf_binadj)
        gf_gdata_new = GraphD_Construt(gf, gf_binadj)
        gf_new_x, gf_new_edge_index = gf_gdata_new.x.to(config.device), gf_gdata_new.edge_index.to(config.device)
        gf_new = self.gcn_re(gf_new_x, gf_new_edge_index)  # b, 10, 4

        fused_f = [shf, gf_new]
        fused_f = torch.cat(fused_f, dim=1)
        fused_f = self.flatten(fused_f)

        return fused_f

class MHG_PPmodel(nn.Module):
    def __init__(self):
        super().__init__()
        '''Shared_Network156_chres  CCTCFsh_Net  CCTCFPWRsh_Net'''
        self.enc = CCTCFPWRsh_Net()

        self.dev_net = Dev_Net()
        self.flatten = nn.Flatten()

        self.output_msw = nn.Sequential(
            nn.Linear(1152 + 6 * 4 + 256, 512),
            # nn.Linear(1152 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.output_rmw = nn.Sequential(
            nn.Linear(1152 + 6 * 4 + 256, 512),
            # nn.Linear(1152 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, diff_x, dev_sh, dev_spv, dev_spr, dev_level, lat, lon, t, cc, tcf, pwr):
        # 128, 2, 2 (hard share)
        shf = self.enc(x, lat, lon, t, cc, tcf, pwr)
        # shf = self.flatten(self.enc(x, lat, lon))

        wind_dev_info, size_dev_info, pp_adj = self.dev_net(diff_x, dev_sh, dev_spv, dev_spr, dev_level)

        # 288+64+2 , 2, 2
        wind_fused_f = [shf, wind_dev_info]
        wind_fused_f = torch.cat(wind_fused_f, dim=1)

        rmw_fused_f = [shf, size_dev_info]
        rmw_fused_f = torch.cat(rmw_fused_f, dim=1)

        msw = self.output_msw(wind_fused_f)
        rmw = self.output_rmw(rmw_fused_f)
        msw = msw[:, 0]
        rmw = rmw[:, 0]

        return msw, rmw
