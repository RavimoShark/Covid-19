import torch
import numpy as np


def create_death_cases(df, scale, pop):
    death = df["deaths"].to_numpy() / pop * scale
    new_cases = df["new_cases"].to_numpy() / pop * scale
    return death, np.cumsum(new_cases)


def make_data(death, cases, init, delta):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = len(cases[init:])
    t = torch.linspace(0.0, size - 1, size)
    data_death = torch.tensor(death[init:]).to(device)
    data_death = data_death.type(torch.FloatTensor).to(device)
    data_cases = torch.tensor(cases[init - delta : -delta]).to(device)
    data_cases = data_cases.type(torch.FloatTensor).to(device)
    return t, data_death, data_cases
