import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint


class IHDS_model(nn.Module):
    def __init__(self, parms, start_confin, end_confin):
        super(IHDS_model, self).__init__()
        self.b1 = torch.nn.Parameter(parms[0])
        self.b2 = torch.nn.Parameter(parms[1])
        self.b3 = torch.nn.Parameter(parms[2])
        self.g = torch.nn.Parameter(parms[3])
        self.nu = torch.nn.Parameter(parms[4])
        self.l = torch.nn.Parameter(parms[5])
        self.start = torch.nn.Parameter(start_confin)
        self.end = torch.nn.Parameter(end_confin)
        self.m = torch.nn.Sigmoid()

    def forward(self, t, y):
        I = y[:, 0]
        H = y[:, 1]
        D = y[:, 2]
        S = y[:, 3]
        R = y[:, 4]
        b = (
            self.b1
            + self.b2 * (self.m(t - self.start))
            + self.b3 * (self.m(t - self.end))
        )
        dS = -b * I * S
        dI = b * I * S - self.g * I - self.nu * I
        dR = self.g * I
        dH = self.nu * I - self.g * H - self.l * H
        dD = self.l * H
        return torch.cat((dI, dH, dD, dS, dR), 0)


class IHD_model(nn.Module):
    def __init__(self, parms, start_confin):
        super(IHD_model, self).__init__()
        self.b1 = torch.nn.Parameter(parms[0])
        self.b2 = torch.nn.Parameter(parms[1])
        self.g = torch.nn.Parameter(parms[2])
        self.nu = torch.nn.Parameter(parms[3])
        self.l = torch.nn.Parameter(parms[4])
        self.start = torch.nn.Parameter(start_confin)
        self.m = torch.nn.Sigmoid()

    def forward(self, t, y):
        I = y[:, 0]
        H = y[:, 1]
        D = y[:, 2]
        b = self.b1 + self.b2 * (self.m(t - self.start))
        dI = b * I - self.g * I - self.nu * I
        dH = self.nu * I - self.g * H - self.l * H
        dD = self.l * H
        return torch.cat((dI, dH, dD), 0)


""" class IHD_fit(nn.Module):
    def __init__(self, parms):
        super(IHD_fit, self).__init__()
        self.b = torch.nn.Parameter(parms[0])
        self.g = torch.nn.Parameter(parms[1])
        self.nu = torch.nn.Parameter(parms[2])
        self.l = torch.nn.Parameter(parms[3])

    def forward(self, t, y):
        I = y[:, 0]
        H = y[:, 1]
        D = y[:, 2]
        S = y[:, 3]
        dS = -b * I * S
        dI = b * I * S - self.g * I - self.nu * I
        dH = self.nu * I - self.g * H - self.l * H
        dD = self.l * H
        return torch.cat((dI, dH, dD, dS), 0)


class IHD_fit_time(nn.Module):
    def __init__(self, parms, time):
        super(IHD_fit_time, self).__init__()
        self.b1 = torch.nn.Parameter(parms[0])
        self.b2 = torch.nn.Parameter(parms[1])
        self.g = torch.nn.Parameter(parms[2])
        self.nu = torch.nn.Parameter(parms[3])
        self.l = torch.nn.Parameter(parms[4])
        self.time = torch.nn.Parameter(time)
        self.m = torch.nn.Sigmoid()

    def forward(self, t, y):
        I = y[:, 0]
        H = y[:, 1]
        D = y[:, 2]
        S = y[:, 3]
        b = self.b1 + self.b2 * self.m(t - self.time)
        dS = -b * I * S
        dI = b * I * S - self.g * I - self.nu * I
        dH = self.nu * I - self.g * H - self.l * H
        dD = self.l * H
        return torch.cat((dI, dH, dD, dS), 0) """


def predic_ode(model, init, t):
    with torch.no_grad():
        true_y = odeint(model, init, t, method="dopri5")
    return true_y.squeeze(1)


def trainig(
    model,
    init,
    t,
    optimizer,
    criterion,
    niters,
    data,
    data_cat={"I": 0, "H": 1, "D": 2, "S": 3},
):
    best_loss = 1000.0
    parms_best = model.parameters()
    ind = np.asarray(list(data_cat.values()), dtype=int)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(ind)
    model.to(device)
    model.train()
    for itr in range(1, niters + 1):
        loss = 0
        optimizer.zero_grad()
        pred_y = odeint(model, init, t)
        loss = criterion(pred_y[:, 0, ind], data)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            parms_best = model.parameters()
        if itr % 10 == 0 or itr == 1:
            print(itr, loss.item(), [p.data.item() for p in model.parameters()])
    return best_loss, list(parms_best)


# to be refactored with mask...
def trainig_hosp(model, init, t, optimizer, criterion, niters, data):
    best_loss = 1000.0
    parms_best = model.parameters()
    for itr in range(1, niters + 1):
        optimizer.zero_grad()
        pred_y = odeint(model, init, t)
        loss = criterion(pred_y[:, 0, 1], data[0]) + criterion(pred_y[:, 0, 2], data[1])
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            parms_best = model.parameters()
        if itr % 10 == 0:
            print(itr, loss.item(), [p.data for p in model.parameters().item()])
    return best_loss, list(parms_best)


def get_best_model(l):
    parms_inf = torch.cat([p.data.unsqueeze(0) for p in l[:-1]], 0)
    time_inf = l[-1].data
    return IHD_fit_time(parms_inf, time_inf)


def get_best_model_simple(l):
    parms_inf = torch.cat([p.data.unsqueeze(0) for p in l], 0)
    return IHD_fit(parms_inf)
