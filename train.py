from __future__ import print_function
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from dataset import FeatureDataset
from modal import SimilarityModule
from modal import LMF
from utils import total, load_pom
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import argparse
import random
import torch.nn as nn
import torch.optim as optim
import csv


# Configs
DEVICE = "cuda:0"
NUM_WORKER = 1
BATCH_SIZE = 64
LR = 1e-3
L2 = 0  # 1e-5
NUM_EPOCH = 100


def prepare_data(text, image, label):    #数据准备(文本、图像、标签)
    nr_index = [i for i, l in enumerate(label) if l == 1]   #判断lable是否为1，如果为1，则生成所有lable为1的下标，并形成数组形式
    text_nr = text[nr_index]    #nr_index里所有标签为1的文本的下标的集合
    image_nr = image[nr_index]      #nr_index里所有标签为1的图像的下标的集合
    fixed_text = copy.deepcopy(text_nr)     #将变量text_nr的值赋给fixed_text，并且两者互不影响，即两者的地址并不相同
    matched_image = copy.deepcopy(image_nr)     #将变量image_nr的值赋给matched_image，并且两者互不影响，即两者的地址并不相同
    unmatched_image = copy.deepcopy(image_nr).roll(shifts=3, dims=0)        #第0维度向下移3次赋值给unmatched_image，打乱顺序
    return fixed_text, matched_image, unmatched_image


def train():        #定义训练集模型
    # ---  Load Config  ---     #初始化一些定义的参数
    device = torch.device(DEVICE)       #表示将构建的张量或者模型分配到相应的设备上，具体则分配到cuda:0上
    num_workers = NUM_WORKER        #初始化为1，工作进程的数量
    batch_size = BATCH_SIZE         #初始化为64，一批样本的大小
    lr = LR         #初始化为1e-3
    l2 = L2         #初始化为0
    num_epoch = NUM_EPOCH         #初始化为100，模型训练迭代的总轮数
    
    # ---  Load Data  ---       #加载数据集，这里都是已经处理好的数据集
    dataset_dir = 'data/twitter'        #目录
    train_set = FeatureDataset(
        "{}/train_text_with_label.npz".format(dataset_dir),
        "{}/train_image_with_label.npz".format(dataset_dir)
    )
    test_set = FeatureDataset(
        "{}/test_text_with_label.npz".format(dataset_dir),
        "{}/test_image_with_label.npz".format(dataset_dir)
    )
    train_loader = DataLoader(#加载的数据集，一批样本的大小，多进程加载的进程数（工作进程的数量），是否将数据打乱
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_loader = DataLoader(#加载的数据集，一批样本的大小，多进程加载的进程数（工作进程的数量），是否将数据打乱
        test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # ---  Build Model & Trainer  ---
    similarity_module = SimilarityModule()      #相似度模型
    similarity_module.to(device)        #将模型移动到指定的设备上

    loss_func_similarity = torch.nn.CosineEmbeddingLoss()       #将交叉熵损失函数作为相似度模型的损失函数

    #这个函数的名字是后面消除梯度的.前面的名称
    optim_task_similarity = torch.optim.Adam(
        similarity_module.parameters(), lr=lr, weight_decay=l2  #给出优化器对象，并给出一个可进行迭代优化的包含了所有参数的列表，指定程序优化的特定选项，这里是学习速率lr和权重衰减weight_decay
    )  # also called task1

    # ---  Model Training  ---
    loss_similarity_total = 0
    loss_detection_total = 0
    best_acc = 0
    for epoch in range(num_epoch):      #生成一个包含0到num_epoch-1之间所有整数的序列，epoch会在循环中以此取到这个序列中的每一个整数值

        similarity_module.train()

        corrects_pre_similarity = 0

        loss_similarity_total = 0

        similarity_count = 0


        for i, (text, image, label) in tqdm(enumerate(train_loader)):       #tqdm是进度条库，可以在python的长循环中添加一个进度提示信息
            batch_size = text.shape[0]      #对于图像来说：image.shape[0]——图片高，image.shape[1]——图片长，image.shape[2]——图片通道数；而对于矩阵来说：，shape[0]：表示矩阵的行数，shape[1]：表示矩阵的列数，shape[-1]表示最后一个维度
            text = text.to(device)          #将模型移到指定的设备上
            image = image.to(device)
            label = label.to(device)

            fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)   #将数据准备中返回的值赋予到这三个变量
            fixed_text.to(device)
            matched_image.to(device)
            unmatched_image.to(device)

            # ---  TASK1 Similarity  ---

            text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(fixed_text, matched_image)       #将相似度函数的返回值赋予到这三个变量
            text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(fixed_text, unmatched_image)
            similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)    #cat是拼接函数，用于拼接向量，dim为拼接的维度，dim为0就是将行拼接起来，dim为1就是将列拼接起来

            #对应文章里的reg的损失函数中标签为1和0的情况
            similarity_label_0 = torch.cat([torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])], dim=0).to(device)
            similarity_label_1 = torch.cat([torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])], dim=0).to(device)#torch.ones/zeros返回一个全为1/0的张量，括号里的是size大小
            
            text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
            image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)

            loss_similarity = loss_func_similarity(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)     #（输入1，输入2，目标函数）
            #一般在遍历epochs的过程中会依次用到一下三个函数
            optim_task_similarity.zero_grad()       #将梯度归零
            loss_similarity.backward()              #反向传播计算得到每个参数的梯度值
            optim_task_similarity.step()            #通过梯度下降执行一步参数更新

            corrects_pre_similarity += similarity_pred.eq(similarity_label_0).sum().item()  #求模型准确的值（准确率）

            # ---  Record  ---

            loss_similarity_total += loss_similarity.item() * (2 * fixed_text.shape[0])     #相似度的总损失

            similarity_count += (2 * fixed_text.shape[0] * 2)                               #计算总共的数量


        loss_similarity_train = loss_similarity_total / similarity_count        #计算训练集相似度的损失函数比值

        acc_similarity_train = corrects_pre_similarity / similarity_count       #计算训练集相似度的准确率比值


        # ---  Test  ---

        acc_similarity_test, loss_similarity_test, cm_similarity = test(similarity_module, test_loader)

        # ---  Output  ---

        print('---  TASK1 Similarity  ---')
        print(
            "EPOCH = %d \n acc_similarity_train = %.3f \n acc_similarity_test = %.3f \n loss_similarity_train = %.3f \n loss_similarity_test = %.3f \n" %
            (epoch + 1, acc_similarity_train, acc_similarity_test, loss_similarity_train, loss_similarity_test)
        )

        print('---  TASK1 Similarity Confusion Matrix  ---')
        print('{}\n'.format(cm_similarity))


#上面是训练集部分，训练出使得损失函数最小，准确率最高的模型参数运用到测试集上

def test(similarity_module, test_loader):
    similarity_module.eval()


    device = torch.device(DEVICE)

    loss_func_similarity = torch.nn.CosineEmbeddingLoss()

    similarity_count = 0

    loss_similarity_total = 0

    similarity_label_all = []

    similarity_pre_label_all = []


    with torch.no_grad():
        for i, (text, image, label) in enumerate(test_loader):
            batch_size = text.shape[0]
            text = text.to(device)
            image = image.to(device)
            label = label.to(device)
            
            fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)
            fixed_text.to(device)
            matched_image.to(device)
            unmatched_image.to(device)

            # ---  TASK1 Similarity  ---

            text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(fixed_text, matched_image)
            text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(fixed_text, unmatched_image)
            similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)
            similarity_label_0 = torch.cat([torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])], dim=0).to(device)
            similarity_label_1 = torch.cat([torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])], dim=0).to(device)

            text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
            image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
            loss_similarity = loss_func_similarity(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)


            # ---  Record  ---

            loss_similarity_total += loss_similarity.item() * (2 * fixed_text.shape[0])

            similarity_count += (fixed_text.shape[0] * 2)


            similarity_pre_label_all.append(similarity_pred.detach().cpu().numpy())

            similarity_label_all.append(similarity_label_0.detach().cpu().numpy())


        loss_similarity_test = loss_similarity_total / similarity_count


        similarity_pre_label_all = np.concatenate(similarity_pre_label_all, 0)

        similarity_label_all = np.concatenate(similarity_label_all, 0)


        acc_similarity_test = accuracy_score(similarity_pre_label_all, similarity_label_all)

        cm_similarity = confusion_matrix(similarity_pre_label_all, similarity_label_all)


    return acc_similarity_test, loss_similarity_test, cm_similarity


def display(mae, corr, mult_acc):
    print("MAE on test set is {}".format(mae))
    print("Correlation w.r.t human evaluation on test set is {}".format(corr))
    print("Multiclass accuracy on test set is {}".format(mult_acc))


def main(options):
    DTYPE = torch.FloatTensor

    # parse the input args
    run_id = options['run_id']
    epochs = options['epochs']
    data_path = options['data_path']
    model_path = options['model_path']
    output_path = options['output_path']
    signiture = options['signiture']
    patience = options['patience']
    output_dim = options['output_dim']

    print("Training initializing... Setup ID is: {}".format(run_id))

    # prepare the paths for storing models and outputs
    model_path = os.path.join(
        model_path, "model_{}_{}.pt".format(signiture, run_id))
    output_path = os.path.join(
        output_path, "results_{}_{}.csv".format(signiture, run_id))
    print("Temp location for models: {}".format(model_path))
    print("Grid search results are in: {}".format(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_set, valid_set, test_set, input_dims = load_pom(data_path)

    params = dict()
    params['audio_hidden'] = [4, 8, 16]
    params['video_hidden'] = [4, 8, 16]
    params['text_hidden'] = [64, 128, 256]
    params['audio_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['video_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['text_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['factor_learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['rank'] = [1, 4, 8, 16]
    params['batch_size'] = [4, 8, 16, 32, 64, 128]
    params['weight_decay'] = [0, 0.001, 0.002, 0.01]

    total_settings = total(params)

    print("There are {} different hyper-parameter settings in total.".format(total_settings))

    seen_settings = set()

    with open(output_path, 'w+') as out:
        writer = csv.writer(out)
        writer.writerow(
            ["audio_hidden", "video_hidden", 'text_hidden', 'audio_dropout', 'video_dropout', 'text_dropout',
             'factor_learning_rate', 'learning_rate', 'rank', 'batch_size', 'weight_decay', 'Best Validation MAE',

             'Confidence accuracy',
             'Passionate accuracy',
             'Pleasant accuracy',
             'Dominant accuracy',
             'Credible accuracy',
             'Vivid accuracy',
             'Expertise accuracy',
             'Entertaining accuracy',
             'Reserved accuracy',
             'Trusting accuracy',
             'Relaxed accuracy',
             'Outgoing accuracy',
             'Thorough accuracy',
             'Nervous accuracy',
             'Persuasive accuracy',
             'Humorous accuracy',

             'Confidence MAE',
             'Passionate MAE',
             'Pleasant MAE',
             'Dominant MAE',
             'Credible MAE',
             'Vivid MAE',
             'Expertise MAE',
             'Entertaining MAE',
             'Reserved MAE',
             'Trusting MAE',
             'Relaxed MAE',
             'Outgoing MAE',
             'Thorough MAE',
             'Nervous MAE',
             'Persuasive MAE',
             'Humorous MAE',

             'Confidence corr',
             'Passionate corr',
             'Pleasant corr',
             'Dominant corr',
             'Credible corr',
             'Vivid corr',
             'Expertise corr',
             'Entertaining corr',
             'Reserved corr',
             'Trusting corr',
             'Relaxed corr',
             'Outgoing corr',
             'Thorough corr',
             'Nervous corr',
             'Persuasive corr',
             'Humorous corr'])

    for i in range(total_settings):

        ahid = random.choice(params['audio_hidden'])
        vhid = random.choice(params['video_hidden'])
        thid = random.choice(params['text_hidden'])
        thid_2 = thid // 2
        adr = random.choice(params['audio_dropout'])
        vdr = random.choice(params['video_dropout'])
        tdr = random.choice(params['text_dropout'])
        factor_lr = random.choice(params['factor_learning_rate'])
        lr = random.choice(params['learning_rate'])
        r = random.choice(params['rank'])
        batch_sz = random.choice(params['batch_size'])
        decay = random.choice(params['weight_decay'])

        # reject the setting if it has been tried
        current_setting = (ahid, vhid, thid, adr, vdr, tdr, factor_lr, lr, r, batch_sz, decay)
        if current_setting in seen_settings:
            continue
        else:
            seen_settings.add(current_setting)

        model = LMF(input_dims, (ahid, vhid, thid), thid_2, (adr, vdr, tdr, 0.5), output_dim, r)
        if options['cuda']:
            model = model.cuda()
            DTYPE = torch.cuda.FloatTensor
        print("Model initialized")
        criterion = nn.L1Loss(size_average=False)
        factors = list(model.parameters())[:3]
        other = list(model.parameters())[3:]
        optimizer = optim.Adam([{"params": factors, "lr": factor_lr}, {"params": other, "lr": lr}],
                               weight_decay=decay)  # don't optimize the first 2 params, they should be fixed (output_range and shift)

        # setup training
        complete = True
        min_valid_loss = float('Inf')
        train_iterator = DataLoader(train_set, batch_size=batch_sz, shuffle=True)
        valid_iterator = DataLoader(valid_set, batch_size=len(valid_set), shuffle=True)
        test_iterator = DataLoader(test_set, batch_size=len(test_set), shuffle=True)
        curr_patience = patience
        for e in range(epochs):
            model.train()
            model.zero_grad()
            avg_train_loss = 0.0
            for batch in train_iterator:
                model.zero_grad()

                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False)
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False)
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False)
                output = model(x_a, x_v, x_t)
                loss = criterion(output, y)
                loss.backward()
                avg_loss = loss.detach().numpy() / float(output_dim)
                avg_train_loss += avg_loss / len(train_set)
                optimizer.step()

            print("Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss))

            # Terminate the training process if run into NaN
            if np.isnan(avg_train_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            model.eval()
            for batch in valid_iterator:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False)
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False)
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False)
                output = model(x_a, x_v, x_t)
                valid_loss = criterion(output, y)
                avg_valid_loss = valid_loss.detach().numpy() / float(output_dim)
            y = y.cpu().data.numpy().reshape(-1, output_dim)

            if np.isnan(avg_valid_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            avg_valid_loss = avg_valid_loss / len(valid_set)
            print("Validation loss is: {}".format(avg_valid_loss))

            if (avg_valid_loss < min_valid_loss):
                curr_patience = patience
                min_valid_loss = avg_valid_loss
                torch.save(model, model_path)
                print("Found new best model, saving to disk...")
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break
            print("\n\n")

        if complete:

            best_model = torch.load(model_path)
            best_model.eval()
            for batch in test_iterator:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False)
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False)
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False)
                output_test = best_model(x_a, x_v, x_t)
                loss_test = criterion(output_test, y)
                test_loss = loss_test.item()
                avg_test_loss = test_loss / float(output_dim)
            output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
            y = y.cpu().data.numpy().reshape(-1, output_dim)

            # these are the needed metrics
            mae = np.mean(np.absolute(output_test - y), axis=0)
            mae = [round(a, 3) for a in mae]
            corr = [round(np.corrcoef(output_test[:, i], y[:, i])[0][1], 3) for i in range(y.shape[1])]
            mult_acc = [round(sum(np.round(output_test[:, i]) == np.round(y[:, i])) / float(len(y)), 3) for i in
                        range(y.shape[1])]

            display(mae, corr, mult_acc)

            results = [ahid, vhid, thid, adr, vdr, tdr, factor_lr, lr, r, batch_sz, decay,
                       min_valid_loss.cpu().data.numpy()]

            results.extend(mult_acc)
            results.extend(mae)
            results.extend(corr)

            with open(output_path, 'a+') as out:
                writer = csv.writer(out)
                writer.writerow(results)


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--run_id', dest='run_id', type=int, default=1)
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=500)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--output_dim', dest='output_dim', type=int, default=16)  # for 16 speaker traits
    OPTIONS.add_argument('--signiture', dest='signiture', type=str, default='')
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=False)
    OPTIONS.add_argument('--data_path', dest='data_path',
                         type=str, default='./data/')
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='results')
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
    train()

