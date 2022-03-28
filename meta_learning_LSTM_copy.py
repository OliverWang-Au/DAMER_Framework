#!/usr/bin/env python3

"""
Trains a 3 layer MLP with MAML on Sine Wave Regression Dataset.
We use the Sine Wave dataloader from the torchmeta package.

Torchmeta: https://github.com/tristandeleu/pytorch-meta
"""

import random
import numpy as np
import torch

from torch import nn, optim
from maml import MAML


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
        adapt_steps=5,
        input_dim = 64,
        hidden_dim=32,
        layer_dim = 3,
        output_dim = 1046
):
    # load the dataset
    #asksets = Sinusoid(num_samples_per_task=2 * shots, num_tasks=num_tasks)
    #dataloader = BatchMetaDataLoader(tasksets, batch_size=tasks_per_batch)

    print("Strat>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # create the model
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    maml = MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = nn.MSELoss(reduction='mean')

    # for each iteration
    for iter, batch in enumerate(dataloader):  # num_tasks/batch_size
        meta_train_loss = 0.0

        # for each task in the batch
        effective_batch_size = batch[0].shape[0]
        for i in range(effective_batch_size):
            learner = maml.clone()

            # divide the data into support and query sets
            train_inputs, train_targets = batch[0][i].float(), batch[1][i].float()
            x_support, y_support = train_inputs[::2], train_targets[::2]
            x_query, y_query = train_inputs[1::2], train_targets[1::2]

            for _ in range(adapt_steps):  # adaptation_steps
                support_preds = learner(x_support)
                support_loss = lossfn(support_preds, y_support)
                learner.adapt(support_loss)

            query_preds = learner(x_query)
            query_loss = lossfn(query_preds, y_query)
            meta_train_loss += query_loss

        meta_train_loss = meta_train_loss / effective_batch_size

        if iter % 200 == 0:
            print('Iteration:', iter, 'Meta Train Loss', meta_train_loss.item())

        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()


if __name__ == '__main__':
    main()