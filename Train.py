import os
import torch.optim
from torch import nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import Set_seed
from Dataset import *
import Config as config
# from NormalCNN import *
# from CCLSTM import *
from MHG_PPNet import *
from MHG_PP_Simp import *
import ssl
import smtplib
from email.header import Header
from email.utils import formataddr
from email.mime.text import MIMEText

def save_model(model, model_output_dir, epoch):
    save_model_file = os.path.join(model_output_dir, "epoch_{}.pth".format(epoch))
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    torch.save(model, save_model_file)

def train_model(model, loss_func, dataset, optimizer, params, epoch):

    model.train()

    cls_batch_loss = 0

    batch_loss = 0
    batch_wind_loss = 0
    batch_rmw_loss = 0

    item = 0
    batch_num = 0

    wind_error = 0
    rmw_error = 0

    for batch, data in enumerate(tqdm(dataset)):
        k8_btemp = data["k8_btemp"]
        k8_btemp = k8_btemp.to(config.device)

        pxh_k8_btemp = data["pxh_k8_btemp"]
        pxh_k8_btemp = pxh_k8_btemp.to(config.device)
        '''cat pxh and now'''
        btemp = torch.cat([pxh_k8_btemp, k8_btemp], dim=1)

        btdiff = data["btdiff"]
        btdiff = btdiff.to(config.device)

        plevel = data["pre_level"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        cat = data["cat"].to(config.device)
        lat = data["lat"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        lon = data["lon"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        t = data['norm_t'].to(torch.float32).to(config.device).unsqueeze(dim=-1)

        ccinfo = data["ccinfo"].to(torch.float32).to(config.device)     # b, 7

        pre_tcf = data["pre_tcf"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        pre_isr = data["pre_isr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        pre_pwr = data["pre_pwr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        pre_prr = data["pre_prr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        pre_pr = torch.cat([pre_tcf, pre_isr, pre_pwr, pre_prr], dim=-1)

        ccnum = data["ccnum"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        moi = data["moi"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        ic = data["bt_ch7_std"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        of = data["ocbt"].to(torch.float32).to(config.device).unsqueeze(dim=-1)

        msw_label = data["msw"]
        msw_label = msw_label.to(config.device)
        rmw_label = data["rmw"]
        rmw_label = rmw_label.to(config.device)

        optimizer.zero_grad()

        msw, rmw = model(btemp, btdiff, moi, ic, of, plevel, lat, lon, t, ccnum, pre_tcf, pre_pwr)

        print("msw_label={}".format(msw_label))
        print("msw={}".format(msw))

        print("rmw_label={}".format(rmw_label))
        print("rmw={}".format(rmw))

        wind_loss = loss_func(msw.float(), msw_label.float())
        rmw_loss = loss_func(rmw.float(), rmw_label.float())

        total_loss = params[0] * wind_loss + params[1] * rmw_loss

        total_loss.backward()
        optimizer.step()

        print("Train Epoch = {} MSW Loss = {}".format(epoch, wind_loss.data.item()))
        print("Train Epoch = {} RMW Loss = {}".format(epoch, rmw_loss.data.item()))

        print("Train Epoch = {} Loss = {}".format(epoch, total_loss.data.item()))
        batch_loss += total_loss.data.item()
        batch_wind_loss += wind_loss.data.item()
        batch_rmw_loss += rmw_loss.data.item()

        # 反归一化看train error
        wind_label_re = msw_label.cpu().detach().numpy() * (170 - 35) + 35
        wind_re = msw.cpu().detach().numpy() * (170 - 35) + 35
        wind_error = wind_error + np.sum(np.abs(wind_re - wind_label_re))

        rmw_label_re = rmw_label.cpu().detach().numpy() * (130 - 5) + 5
        rmw_re = rmw.cpu().detach().numpy() * (130 - 5) + 5
        rmw_error = rmw_error + np.sum(np.abs(rmw_re - rmw_label_re))

        batch_num += 1
        item += len(rmw_re)

    print("Train Epoch = {} mean loss = {} ".format(epoch, batch_loss / batch_num))
    print("Train Epoch = {} msw error = {} ".format(epoch, wind_error / item))
    print("Train Epoch = {} rmw error = {} ".format(epoch, rmw_error / item))

    return batch_loss / batch_num, cls_batch_loss / batch_num, wind_error / item, rmw_error / item

def valid_model(model, loss_func, dataset, epoch):

    model.eval()

    batch_loss = 0
    cls_batch_loss = 0

    batch_num = 0
    item = 0

    wind_error = 0
    rmw_error = 0

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

            ccinfo = data["ccinfo"].to(torch.float32).to(config.device)  # b, 7

            msw_label = data["msw"]
            msw_label = msw_label.to(config.device)
            rmw_label = data["rmw"]
            rmw_label = rmw_label.to(config.device)

            plevel = data["pre_level"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            cat = data["cat"].to(config.device)
            lat = data["lat"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            lon = data["lon"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            t = data['norm_t'].to(torch.float32).to(config.device).unsqueeze(dim=-1)

            pre_tcf = data["pre_tcf"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_isr = data["pre_isr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_pwr = data["pre_pwr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_prr = data["pre_prr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_pr = torch.cat([pre_tcf, pre_isr, pre_pwr, pre_prr], dim=-1)

            ccnum = data["ccnum"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            moi = data["moi"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            ic = data["bt_ch7_std"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            of = data["ocbt"].to(torch.float32).to(config.device).unsqueeze(dim=-1)

            msw, rmw = model(btemp, btdiff, moi, ic, of, plevel, lat, lon, t, ccnum, pre_tcf, pre_pwr)

            print("wind_label={}".format(msw_label))
            print("msw={}".format(msw))

            print("rmw_label={}".format(rmw_label))
            print("RMW={}".format(rmw))

            wind_loss = loss_func(msw.float(), msw_label.float())
            rmw_loss = loss_func(rmw.float(), rmw_label.float())

            total_loss = rmw_loss + wind_loss

            print("Valid Epoch = {} msw Loss = {}".format(epoch, wind_loss.data.item()))
            print("Valid Epoch = {} rmw Loss = {}".format(epoch, rmw_loss.data.item()))

            print("Valid Epoch = {} Loss = {}".format(epoch, total_loss.data.item()))
            batch_loss += total_loss.data.item()

            # 标签和估计结果反归一化后比较误差
            wind_label_re = msw_label.cpu().detach().numpy() * (170 - 35) + 35
            wind_re = msw.cpu().detach().numpy() * (170 - 35) + 35
            wind_error = wind_error + np.sum(np.abs(wind_re - wind_label_re))

            rmw_label_re = rmw_label.cpu().detach().numpy() * (130 - 5) + 5
            rmw_re = rmw.cpu().detach().numpy() * (130 - 5) + 5
            rmw_error = rmw_error + np.sum(np.abs(rmw_re - rmw_label_re))

            batch_num += 1
            item += len(rmw_re)

    print("Valid Epoch = {} mean loss = {} ".format(epoch, batch_loss / batch_num))
    print("Valid Epoch = {} msw error = {} ".format(epoch, wind_error / item))
    print("Valid Epoch = {} rmw error = {} ".format(epoch, rmw_error / item))

    return batch_loss / batch_num, cls_batch_loss / batch_num, wind_error / item, rmw_error / item

def train(model, Estim_Loss, optimizer, params, epochs):
    train_transform = None
    valid_transform = None
    train_dataset = Findpxh_Dataset_k_pr(config.train_k8_path, 3, None, config.data_format)
    valid_dataset = Findpxh_Dataset_k_pr(config.valid_k8_path, 3, None, config.data_format)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )

    '''
    训练模型
    '''
    start_epoch = 0
    train_loss_array = np.zeros(epochs + 1)
    valid_loss_array = np.zeros(epochs + 1)
    train_cls_loss_array = np.zeros(epochs + 1)
    valid_cls_loss_array = np.zeros(epochs + 1)
    # 强度
    train_wind_error_array = np.zeros(epochs + 1)
    valid_wind_error_array = np.zeros(epochs + 1)
    # 风圈RMW
    train_rmw_error_array = np.zeros(epochs + 1)
    valid_rmw_error_array = np.zeros(epochs + 1)

    for epoch in range(start_epoch + 1, epochs + 1):
        train_epoch_loss, train_cls_loss, train_wind_error, train_rmw_error = \
            train_model(model, Estim_Loss, train_dataloader, optimizer, params, epoch)
        valid_epoch_loss, valid_cls_loss, valid_wind_error, valid_rmw_error = \
            valid_model(model, Estim_Loss, valid_dataloader, epoch)

        train_loss_array[epoch] = train_epoch_loss
        valid_loss_array[epoch] = valid_epoch_loss

        train_cls_loss_array[epoch] = train_cls_loss
        valid_cls_loss_array[epoch] = valid_cls_loss

        train_wind_error_array[epoch] = train_wind_error
        valid_wind_error_array[epoch] = valid_wind_error

        train_rmw_error_array[epoch] = train_rmw_error
        valid_rmw_error_array[epoch] = valid_rmw_error

        # 模型保存
        if epoch % config.save_model_iter == 0:
            save_model(model, config.model_output_dir, epoch)

    # 绘制loss
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
    train_loss, = plt.plot(np.arange(1, epochs + 1), train_loss_array[1:], 'r')
    val_loss, = plt.plot(np.arange(1, epochs + 1), valid_loss_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('Estimation loss')
    plt.title("train/valid loss vs epoch")
    # 添加图例
    ax_loss.legend(handles=[train_loss, val_loss], labels=['train_epoch_loss', 'val_epoch_loss'],
                         loc='best')
    fig_loss.savefig(config.save_fig_dir + 'loss.png')
    plt.close(fig_loss)

    fig_seg, ax_seg = plt.subplots(figsize=(12, 8))
    train_seg, = plt.plot(np.arange(1, epochs + 1), train_cls_loss_array[1:], 'r')
    val_seg, = plt.plot(np.arange(1, epochs + 1), valid_cls_loss_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('cls loss')
    plt.title("train/valid cls_loss vs epoch")
    # 添加图例
    ax_seg.legend(handles=[train_seg, val_seg], labels=['train_epoch_cls_loss', 'val_epoch_cls_loss'],
                   loc='best')
    fig_seg.savefig(config.save_fig_dir + 'cls_loss.png')
    plt.close(fig_seg)

    # 绘制强度误差
    fig_int_error, ax_int_error = plt.subplots(figsize=(12, 8))
    train_wind_error, = plt.plot(np.arange(1, epochs + 1),
                                                      train_wind_error_array[1:], 'r')
    valid_wind_error, = plt.plot(np.arange(1, epochs + 1),
                                       valid_wind_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('Intensity Error')
    plt.title("train/valid error vs epoch")
    # 添加图例
    ax_int_error.legend(handles=[train_wind_error,
                              valid_wind_error],
                     labels=['train_error', 'valid_error'],
                     loc='best')
    fig_int_error.savefig(config.save_fig_dir + 'int_error.png')
    plt.close(fig_int_error)

    # 绘制RMW误差
    fig_rmw_error, ax_rmw_error = plt.subplots(figsize=(12, 8))
    train_rmw_error, = plt.plot(np.arange(1, epochs + 1),
                                 train_rmw_error_array[1:], 'r')
    valid_rmw_error, = plt.plot(np.arange(1, epochs + 1),
                                 valid_rmw_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('RMW Error')
    plt.title("train/valid error vs epoch")
    # 添加图例
    ax_rmw_error.legend(handles=[train_rmw_error,
                                 valid_rmw_error],
                        labels=['train_error', 'valid_error'],
                        loc='best')
    fig_rmw_error.savefig(config.save_fig_dir + 'rmw_error.png')
    plt.close(fig_rmw_error)

if __name__ == '__main__':
    Set_seed.setup_seed(0)
    '''MHG_PPmodel_Simp'''
    model = MHG_PPmodel().to(config.device)

    params = [1, 1, 1]
    Estim_Loss = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train(model, Estim_Loss, optimizer, params, config.epochs)
