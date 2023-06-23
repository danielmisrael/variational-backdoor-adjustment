import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision
import torch.distributions as td

def add_noise(x):
    noise = x.new().resize_as_(x).uniform_()
    x = x * 255 + noise
    x = x / 256
    return x

class MNISTCausalDataset(Dataset):
    def __init__(self, do_x=None, do_z=None, train=True):
        
        torch.manual_seed(0)

        mnist = torchvision.datasets.MNIST(root='../data', train=train, download=True)

        self.dataset = mnist
        self.do_x = do_x
        self.do_z = do_z

        self.classes = {}

        for i in range(10):
            idx = (self.dataset.targets==i) 
            self.classes[i] = self.dataset.data[idx]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, n):
        image, label = self.dataset.__getitem__(n)
        Z = image
        
        if self.do_z is not None:
            arr = self.classes[self.do_z].numpy()
            idx = np.random.randint(len(arr))
            label = self.do_z
            Z = Image.fromarray(arr[idx])
        
        if self.do_x is not None:
            image_num = self.do_x
        else:
            image_num = (label + 1) % 10
            
        arr = self.classes[image_num].numpy()
        idx = np.random.randint(len(arr))
        X = Image.fromarray(arr[idx])

        left = (transforms.RandomAffine(30)(Z.copy())).resize((14, 28))
        right = (transforms.RandomAffine(30)(X.copy())).resize((14, 28))

        new_image = Image.new('L', (28, 28))
        new_image.paste(left, (0, 0))
        new_image.paste(right, (14, 0))
        
        X = add_noise(transforms.ToTensor()(X)).flatten()
        
        Y = add_noise(transforms.ToTensor()(new_image)).flatten()

        Z = add_noise(transforms.ToTensor()(Z)).flatten()
        

        return X, Y, Z
    


class BinaryMNISTCausalDataset(Dataset):
    def __init__(self, do_x=None, do_z=None, test=False):
        
        torch.manual_seed(0)
        np.random.seed(0)


        mnist = torchvision.datasets.MNIST(root='../data', train=True, download=True)
        train_set, test_set = torch.utils.data.random_split(mnist, [50000, 10000])

        self.dataset = train_set
        if test:
            self.dataset = test_set
        
        self.do_x = do_x
        self.do_z = do_z
        self.classes = {}
        
        for i in range(10):
            idx = (self.dataset.dataset.targets==i) 
            self.classes[i] = self.dataset.dataset.data[idx]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, n):
        image, label = self.dataset.__getitem__(n)
        Z = image
        
        if self.do_z is not None:
            arr = self.classes[self.do_z].numpy()
            idx = np.random.randint(len(arr))
            label = self.do_z
            Z = Image.fromarray(arr[idx])
        
        if self.do_x is not None:
            image_num = self.do_x
        else:
            image_num = (label + 1) % 10
            
        arr = self.classes[image_num].numpy()
        idx = np.random.randint(len(arr))
        X = Image.fromarray(arr[idx])

        left = (transforms.RandomAffine(30)(Z.copy())).resize((14, 28))
        right = (transforms.RandomAffine(30)(X.copy())).resize((14, 28))

        new_image = Image.new('L', (28, 28))
        new_image.paste(left, (0, 0))
        new_image.paste(right, (14, 0))
        
        
        Y = transforms.ToTensor()(new_image)
        Y = td.Bernoulli(Y).sample().flatten()
        
        Z = transforms.ToTensor()(Z)
        Z = td.Bernoulli(Z).sample().flatten()
        
        X = transforms.ToTensor()(X)
        X = td.Bernoulli(X).sample().flatten()
        
        return X, Y, Z
    
 
    
 