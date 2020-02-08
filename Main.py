import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.optim as optim

from Loss import ContrastiveLoss
from SimaseDataset import SiameseMNIST
from SimaseNetwork import SimaseNet


train_dataset = MNIST('../data/MNIST', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ]))

test_dataset = MNIST('../data/MNIST', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor()
                            ]))

def splot(img1,img2):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(transforms.ToPILImage()(img1))
    axarr[1].imshow(transforms.ToPILImage()(img2))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


siamese_train_dataset = SiameseMNIST(train_dataset)
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size = 64, shuffle=True)

simase_test_dataset = SiameseMNIST(test_dataset)
siamese_test_loader = torch.utils.data.DataLoader(simase_test_dataset, batch_size = 1, shuffle=True)

model = SimaseNet()
criterion = ContrastiveLoss(1)
optimizer = optim.Adam(model.parameters(), 1e-2)

train = True

if train:
    # training
    model.train()
    losses = []
    total_loss = 0
    epochs = 10
    iteration = 0
    counter = []
    chunk = 100



    for epoch in range(epochs):

        for batch_idx, data in enumerate(siamese_train_loader):

            img1, img2, target = data

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, target)
            loss.backward()
            optimizer.step()


            if batch_idx % chunk == 0:
                print("Epoch : ", epoch, "Batch : " , batch_idx, "loss : ", loss.item())
                iteration += chunk
                counter.append(iteration)
                losses.append(loss.item())
                total_loss += loss.item()

        show_plot(counter,losses)
        print("Epoch", epoch, "loss ", total_loss /len(data))
        total_loss = 0.0

    torch.save(model, 'simase.pth')


if not train:

    model = torch.load('simase.pth')
    model.eval()

    for batch_idx, data in enumerate(siamese_test_loader):
        img1, img2, target = data
        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, target)
        print(loss)
        splot(img1[0],img2[0])













