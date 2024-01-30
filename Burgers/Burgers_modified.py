import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import os
import time

from pyDOE import lhs
from scipy.interpolate import griddata


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1], bias=False))
        self.model = nn.Sequential(*modules)

    def forward(self, X):
        return self.model(X)

class PhysicsInformedNN(nn.Module):
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):
        super(PhysicsInformedNN, self).__init__()

        self.lb = torch.Tensor(lb)
        self.ub = torch.Tensor(ub)

        self.x_u = torch.Tensor(X_u[:, 0:1])
        self.t_u = torch.Tensor(X_u[:, 1:2])

        self.x_f = torch.Tensor(X_f[:, 0:1])
        self.t_f = torch.Tensor(X_f[:, 1:2])

        self.u = torch.Tensor(u)

        self.layers = layers
        self.nu = nu

        # Initialize MLP
        self.model_u = MLP(layers)

    def neural_net(self, X):
        return self.model_u(X)

    def net_u(self, x, t):
        X = torch.cat([x, t], 1)
        u = self.neural_net(X)
        return u

    def net_f(self, x, t):
        x = torch.Tensor(x).to(self.lb.device).requires_grad_(True)
        t = torch.Tensor(t).to(self.lb.device).requires_grad_(True)

        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        f = u_t + u * u_x - self.nu * u_xx
        return f

    def loss(self, x_u, t_u, u, x_f, t_f):
        u_pred = self.net_u(x_u, t_u)
        f_pred = self.net_f(x_f, t_f)
        loss = torch.mean((u - u_pred)**2) + torch.mean(f_pred**2)
        return loss

    def forward(self, x_u, t_u, u, x_f, t_f):
        return self.loss(x_u, t_u, u, x_f, t_f)

    def predict(self, X_star):
        u_star = self.net_u(X_star[:, 0:1], X_star[:, 1:2]).detach().numpy()
        f_star = self.net_f(X_star[:, 0:1], X_star[:, 1:2]).detach().numpy()
        return u_star, f_star


if __name__ == '__main__':

    n_epoch = 6000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'./model'
    train_info_path = r'./'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    nu = 0.01/torch.pi
    N_u = 100
    N_f = 1000
    layers = (2, 100, 100, 100, 1)

    data = scipy.io.loadmat('/Users/sunguiquan/PycharmProjects/Pytorch/PINNs/data/burgers_shock.mat')

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T  # calculate real part value

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # set domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = Exact[:, -1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub - lb) * lhs(2, N_f)  # 采样器进行采样，生成二维采样矩阵，每一行即为一个采样点
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    # randomly select a subset(子集) of existing training dataset
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    x_u = torch.Tensor(X_u_train[:, 0:1])
    t_u = torch.Tensor(X_u_train[:, 1:2])

    x_f = torch.Tensor(X_f_train[:, 0:1])
    t_f = torch.Tensor(X_f_train[:, 1:2])

    pinn = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)
    pinn = pinn.to(device)

    # adam training
    optimizer_adam = optim.Adam(pinn.parameters(), lr=1e-3)
    loss_log_adam = []
    with open(train_info_path + 'train_info.txt', 'w') as f:
        f.write('Training by Adam:\n')
    start_time = time.time()

    for epoch in range(n_epoch):
        pinn.zero_grad()
        loss = pinn.loss(x_u, t_u, torch.Tensor(u_train), x_f, t_f)
        loss.backward()
        optimizer_adam.step()

        loss_log_adam.append(loss.item())
        if (epoch + 1) % 100 == 0:
            info = f'Epoch # {epoch + 1:4d}/{n_epoch}\ttime:{time.time() - start_time:.1f}\t' + f'loss:{loss.item():.2e}'
            with open(train_info_path + 'train_info_Burgers.txt', 'a') as f:
                f.write(info + '\n')
            print(info)

    # plot loss graph
    plt.figure(figsize=(8, 6))
    plt.subplot(111)
    plt.plot(loss_log_adam , label='$loss$')
    plt.xlabel('epochs')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig('loss_log.png', dpi=100)
    plt.show()


    # use trained model to predict
    X_star = torch.Tensor(np.hstack((X.flatten()[:, None], T.flatten()[:, None]))).to(device)
    u_pred, _ = pinn.predict(X_star)

    # meshing predict result
    U_pred = griddata(X_star.cpu().numpy(), u_pred.flatten(), (X, T), method='cubic')

    # set size of graph
    plt.figure(figsize=(21, 5))

    # exact solution
    plt.subplot(1, 3, 1)
    plt.pcolor(T, X, Exact, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('True Solution')
    plt.tight_layout()

    # predicted solution
    plt.subplot(1, 3, 2)
    plt.pcolor(T, X, U_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Predicted Solution')
    plt.tight_layout()

    # error
    plt.subplot(1, 3, 3)
    error = np.abs(Exact - U_pred)
    plt.pcolor(T, X, error, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Absolute error')
    plt.tight_layout()

    plt.savefig('Burgers_result.png', dpi=100)
    plt.show()


