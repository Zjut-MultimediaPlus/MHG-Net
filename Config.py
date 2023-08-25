# train
epochs = 375
batch_size = 1
device = 'cuda:3'  # cpu or 'cuda:0'

train_npy_path251 = '/MHG-Net/train_156/'
valid_npy_path251 = '/MHG-Net/valid_156/'
train_npy_path_diff = 'MHG-Net/train_diff/'
valid_npy_path_diff = '/MHG-Net/valid_diff/'

num_workers = 4  # 加载数据集线程并发数

best_loss = 0.005  # 当loss小于等于该值会保存模型

save_model_iter = 25  # 每多少次保存一份模型

model_output_dir = '/MHG-Net/'

# predict
predict_model = '/MHG-Net/epoch_375.pth'
predict_npy_path156 = 'MHG-Net/test_156/'
predict_npy_path_diff = 'MHG-Net/test_diff/'


data_format = 'npy'
