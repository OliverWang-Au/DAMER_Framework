#!/usr/bin/env python3

"""
Trains a LSTM with MAML on four cities Dataset.

"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn, optim
from maml import MAML
import numpy
from metrics import Metrics
from itertools import cycle

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # print(out.shape)
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!

        out = self.fc(out[:, -1, :])

        h_n_last = hn[-1]

        # print(out.shape)
        # out.size() --> 100, 10
        return out

def main(
        shots=10,
        tasks_per_batch=16,
        num_tasks=160000,
        adapt_lr=0.01,
        meta_lr=0.001,
        ft_lr = 0.001,
        adapt_steps=5,
        input_dim = 200,
        hidden_dim=32,
        layer_dim = 3,
        output_dim_poi_level = 413,
        tasks_length = 4,
        epchos = 500,
        epcho_ft = 400,
        batch_size = 64

):
    # load the dataset
    train_dataset_sup_cal = torch.load('./cal_dl_sup')
    train_dataset_qry_cal = torch.load('./cal_dl_qry')

    train_dataset_sup_pho = torch.load('./pho_dl_sup')
    train_dataset_qry_pho = torch.load('./pho_dl_qry')

    train_dataset_sup_sin = torch.load('./sin_dl_sup')
    train_dataset_qry_sin = torch.load('./sin_dl_qry')

    train_dataset_sup_newy = torch.load('./newy_dl_sup')
    train_dataset_qry_newy = torch.load('./newy_dl_qry')

    dl_sup_cal = DataLoader(train_dataset_sup_cal, batch_size=batch_size, shuffle=True, drop_last= True)
    dl_qry_cal = DataLoader(train_dataset_qry_cal, batch_size=batch_size, shuffle=True, drop_last= True)

    dl_sup_pho = DataLoader(train_dataset_sup_pho, batch_size=batch_size, shuffle=True, drop_last= True)
    dl_qry_pho = DataLoader(train_dataset_qry_pho, batch_size=batch_size, shuffle=True, drop_last= True)

    dl_sup_sin = DataLoader(train_dataset_sup_sin, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_qry_sin = DataLoader(train_dataset_qry_sin, batch_size=batch_size, shuffle=True, drop_last=True)

    dl_sup_newy = DataLoader(train_dataset_sup_newy, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_qry_newy = DataLoader(train_dataset_qry_newy, batch_size=batch_size, shuffle=True, drop_last=True)

    dl_sup_c1,dl_qry_c1 = dl_sup_cal,dl_qry_cal
    dl_sup_c2, dl_qry_c2 = dl_sup_pho, dl_qry_pho
    dl_sup_c3, dl_qry_c3 = dl_sup_sin, dl_qry_sin
    dl_sup_c4, dl_qry_c4 = dl_sup_newy, dl_qry_newy
    ########################################################################################################
    #Fine Tune dataset
    ft_train_cal = torch.load('./cal_dl_train_ft')
    ft_test_cal = torch.load('./cal_dl_test_ft')

    dl_ft_train_cal = DataLoader(ft_train_cal, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_ft_test_cal = DataLoader(ft_test_cal, batch_size=batch_size, shuffle=True, drop_last=True)


    ########################################################################################################
    #for step, (batchX, batchY) in enumerate(dl_sup_cal):
        #print('| Step: ', step, '| batch x: ',
              #len(batchX.numpy()), '| batch y: ', len(batchY.numpy()))
        #print(batchX.shape, batchY.shape)

    ###################################################
    print("Strat>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # create the model
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim_poi_level)
    maml = MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)
    print(maml)
    opt = optim.Adam(maml.parameters(), meta_lr)
    #lossfn = nn.MSELoss(reduction='mean')
    lossfn = nn.CrossEntropyLoss()

    #cal_train_data = zip(dl_sup_cal, dl_qry_cal)
    #pho_train_data = zip(dl_sup_pho, dl_qry_pho)
    #tasks = zip(zip(dl_sup_cal, dl_qry_cal), zip(dl_sup_pho, dl_qry_pho))
    print('LSTM parameters: ')
    for name, param in model.named_parameters():
        print(name)

    print('MAML parameters: ')
    for name, param in maml.named_parameters():
        print(name, param)
        break

    """
    Meta-learning with lstm Train Progress:
    """
    for epcho in range(epchos):
        tasks = zip(zip(dl_sup_c1, dl_qry_c1), zip(dl_sup_c2, dl_qry_c2), zip(dl_sup_c3, dl_qry_c3), zip(dl_sup_c4, dl_qry_c4))
        #print ('Epcho: ', epcho)
        #retrive data from four tasks
        meta_loss=[]
        for step, iteration_batch_task in enumerate(tasks):
            meta_train_loss = 0.0
            for i, (support, query) in enumerate(iteration_batch_task):

                learner = maml.clone()

                for _ in range(adapt_steps):  # adaptation_steps
                    support_preds = learner(support[0])
                    support_loss = lossfn(support_preds, support[1].long())
                    learner.adapt(support_loss)
                    #print('support_loss', support_loss)

                query_preds = learner(query[0])
                query_loss = lossfn(query_preds, query[1].long())
                meta_train_loss += query_loss
                #print('query_loss', query_loss)

            meta_train_loss = (meta_train_loss / tasks_length)
            #print('meta_train_loss', meta_train_loss)
            meta_loss.append(meta_train_loss.item())
            #if epcho % 10 == 0:
                #print('Epcho:', epcho, 'Meta Train Loss: ', meta_train_loss.item())

            opt.zero_grad()
            meta_train_loss.backward()
            opt.step()
        print('Epcho: ', epcho, 'Meta-average-loss: ', numpy.mean(meta_loss))

        #########################################################################

    print('LSTM parameters after training: ')
    for name, param in learner.named_parameters():
        print(name, param)
        break

    print('MAML parameters after training: ')
    for name, param in maml.named_parameters():
        print(name, param)
        break


    """ 
    Fine Tune Stage 
    """
    model_ft = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim_poi_level)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=ft_lr)
    lossfn_ft = nn.CrossEntropyLoss()

    print('Model_ft parameters_before copy: ')
    for name, param in model_ft.named_parameters():
        print(name, param)
        break

    print('COPY>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # original saved file with DataParallel
    state_dict = maml.state_dict()  # 加载之前的模型。
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    # load params
    model_ft.load_state_dict(new_state_dict)  # 从新加载这个模型。


    print('Model_ft parameters_after copy: ')
    for name, param in model_ft.named_parameters():
        print(name, param)
        break

    total_loss = []
    for epcho in range(epcho_ft):
        loss_all=[]
        for i, (input, label) in enumerate(dl_ft_train_cal):
            optimizer_ft.zero_grad()
            pred = model_ft(input)
            loss = lossfn_ft(pred, label.long())
            loss.backward()
            optimizer_ft.step()
            loss_all.append(loss.item())
        l = numpy.mean(loss_all)

        if epcho %20 ==0:
            print('Epcho: ', epcho, 'Fine_Tune_Average_Loss: ', l)
            total_loss.append(l)
            metric = Metrics()

            hit5_socre_all = []
            hit10_socre_all = []
            ndcg5_socre_all = []
            ndcg10_socre_all = []

            for i, (input, label) in enumerate(dl_ft_test_cal):
                pred = model_ft(input)
                """
                Hit@5 and Hit@10
                """
                hit_socre5 = metric.hits_score(label, pred, k =5)
                hit_socre10 = metric.hits_score(label, pred, k=10)
                hit5_socre_all.append(hit_socre5)
                hit10_socre_all.append(hit_socre10)

                """
                nDCG@5 and nDCG@10
                """
                ndcg_socre5 = metric.ndcg_score(label, pred, k=5)
                ndcg_socre10 = metric.ndcg_score(label, pred, k=10)
                ndcg5_socre_all.append(ndcg_socre5)
                ndcg10_socre_all.append(ndcg_socre10)

            print('Hit@5: ',numpy.mean(hit5_socre_all),'Hit@10: ', numpy.mean(hit10_socre_all))
            print('nDCG@5: ', numpy.mean(ndcg5_socre_all), 'nDCG@10: ', numpy.mean(ndcg10_socre_all))


if __name__ == '__main__':
    main()

