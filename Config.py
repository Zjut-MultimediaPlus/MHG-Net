# train
epochs = 200
batch_size = 16
device = 'cuda:0'  # cpu or 'cuda:0'
id_dim = 16

train_npy_path = '/TCdata/train/'
valid_npy_path = '/TCdata/valid/'
predict_npy_path = '/TCdata/test/'

k8_sta_path = "/opt/data/private/norm_data_npy/Full2015_2023_Dataets/newk8_norm_values_split.pkl"
labels_path = "/TCdata/NameTime_Idx_BST.pkl"
p12hpr_pth = '/opt/data/private/Auxiliay_StatisData/2015_2023/pre12h_labels.pkl'

cc_pth = '/opt/data/private/Auxiliay_StatisData/2015_2023/btemp/btemp_cc.pkl'
ic_pth = '/opt/data/private/Auxiliay_StatisData/2015_2023/btemp/btemp_ic.pkl'
of_pth = '/opt/data/private/Auxiliay_StatisData/2015_2023/btemp/btemp_of.pkl'

train_k8_path = '/TCdata/k89_4ch1/train/'
valid_k8_path = '/TCdata/k89_4ch1/valid/'
predict_k8_path = '/TCdata/k89_4ch1/test/'

model_output_dir = '/opt/data/private/model/MHG_PP/'
predict_model = '/opt/data/private/model/MHG_PP/epoch_200.pth'
save_fig_dir = '/opt/data/private/model/MHG_PP/exp_img/'

num_workers = 4  # 加载数据集线程并发数
best_loss = 0.005  # 当loss小于等于该值会保存模型
save_model_iter = 25  # 每多少次保存一份模型

data_format = 'npy'