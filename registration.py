import sys

import math

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt

print('\npython started:\n')

FOCAL_LENGTH = 4.73e-3
PIXEL_LENGTH = 6.43e-3 / 4000

ROWS = float(sys.argv[1])
COLS = float(sys.argv[2])

y = (ROWS / 2 - float(sys.argv[3])) * PIXEL_LENGTH
x = (float(sys.argv[4]) - COLS / 2) * PIXEL_LENGTH
oa = np.array([x, y, -FOCAL_LENGTH])

y = (ROWS / 2 - float(sys.argv[5])) * PIXEL_LENGTH
x = (float(sys.argv[6]) - COLS / 2) * PIXEL_LENGTH
ob = np.array([x, y, -FOCAL_LENGTH])

y = (ROWS / 2 - float(sys.argv[7])) * PIXEL_LENGTH
x = (float(sys.argv[8]) - COLS / 2) * PIXEL_LENGTH
oc = np.array([x, y, -FOCAL_LENGTH])

aob = math.acos(np.dot(oa, ob) / (np.linalg.norm(oa) * np.linalg.norm(ob)))
boc = math.acos(np.dot(ob, oc) / (np.linalg.norm(ob) * np.linalg.norm(oc)))
coa = math.acos(np.dot(oc, oa) / (np.linalg.norm(oc) * np.linalg.norm(oa)))

print('[camera] aob:', aob * 180 / math.pi)
print('[camera] boc:', boc * 180 / math.pi)
print('[camera] coa:', coa * 180 / math.pi)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

camera_angles = torch.tensor([aob, boc, coa], dtype=torch.float, device=device)

A = torch.tensor([0, 0, 0], dtype=torch.float, device=device)
B = torch.tensor([27.5e-2, 5.8e-2, 0], dtype=torch.float, device=device)
C = torch.tensor([21e-2, -8.3e-2, 0], dtype=torch.float, device=device)

scale = np.linalg.norm(B.numpy()) / np.linalg.norm(ob - oa)
ox = torch.tensor([-scale * oa[0]], requires_grad=True, dtype=torch.float, device=device)
oy = torch.tensor([-scale * oa[1]], requires_grad=True, dtype=torch.float, device=device)
oz = torch.tensor([scale * FOCAL_LENGTH], dtype=torch.float, device=device)
o = torch.cat([ox, oy, oz])

n_epochs = 1000

loss_fn = nn.MSELoss(reduction='mean')

optimizer = optim.SGD([ox, oy], lr=0.1, momentum=0.9)

loss_values = []
for epoch in range(n_epochs):
    o = torch.cat([ox, oy, oz])
    oa = A - o
    ob = B - o
    oc = C - o
    aOb = torch.acos(torch.dot(oa, ob) / (torch.norm(oa) * torch.norm(ob)))
    bOc = torch.acos(torch.dot(ob, oc) / (torch.norm(ob) * torch.norm(oc)))
    cOa = torch.acos(torch.dot(oc, oa) / (torch.norm(oc) * torch.norm(oa)))
    real_angles = torch.stack([aOb, bOc, cOa])

    loss = loss_fn(camera_angles, real_angles)
    loss.backward()
    loss_values.append(loss.data.numpy())

    optimizer.step()
    optimizer.zero_grad()

    if epoch == n_epochs - 1:
        print('\nloss:', loss.data.numpy())

print('\nPredicted camera position:')
print(' o :', o.data.numpy())
print('|o|:', torch.norm(o).data.numpy())
print('')

oa = (A - o).detach().numpy()
ob = (B - o).detach().numpy()
oc = (C - o).detach().numpy()
aOb = math.acos(np.dot(oa, ob) / (np.linalg.norm(oa) * np.linalg.norm(ob)))
bOc = math.acos(np.dot(ob, oc) / (np.linalg.norm(ob) * np.linalg.norm(oc)))
cOa = math.acos(np.dot(oc, oa) / (np.linalg.norm(oc) * np.linalg.norm(oa)))
print('[real] AOB:', aOb * 180 / math.pi)
print('[real] BOC:', bOc * 180 / math.pi)
print('[real] COA:', cOa * 180 / math.pi)

plt.plot(loss_values)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss.png')
