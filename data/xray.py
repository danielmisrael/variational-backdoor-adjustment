from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision
from medmnist import ChestMNIST
from PIL import Image, ImageDraw

def add_noise(x):
    noise = x.new().resize_as_(x).uniform_()
    x = x * 255 + noise
    x = x / 256
    return x


class XrayCausalDataset(Dataset):
    def __init__(self, split='train', download=False):
        
        self.base_dataset = ChestMNIST(split=split, download='True')
        mnist = torchvision.datasets.MNIST(root='../data', train=split, download=True)
        self.zeros = mnist.data[mnist.targets==0]
        np.random.seed(seed=0)

        if download:
            self.create_data(split)

        self.split = split

    def create_data(self, split):
        for i in range(len(self.base_dataset)):

            LEFT = 0.5
            ACCURACY = 0.6
            EFFICACY = 0.7

            img = self.base_dataset.__getitem__(i)[0]
            zero = self.zeros[np.random.randint(0, len(self.zeros))]

            is_left = np.random.random() < LEFT

            left = np.random.randint(low=8, high=11)
            right = np.random.randint(low=17, high=20)
            center_x = left if is_left else right


            center_y = np.random.randint(low=10, high=12)
            height = np.random.randint(low=3, high=4)
            width = np.random.randint(low=3, high=4)

            tumor = Image.new('L', img.size, 0)
            draw = ImageDraw.Draw(tumor)
            draw.ellipse((center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2), fill=255)
            patient = Image.blend(img, tumor, 0.2)

            draw = ImageDraw.Draw(tumor)
            draw.ellipse((center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2), fill=255)


            noise1 = np.random.randint(low=-1, high=1)
            noise2 = np.random.randint(low=-1, high=1)
            treatment = Image.new('L', img.size, 0)


            is_accurate = np.random.random() < ACCURACY
            if is_accurate:
                pos_x = left if is_left else right
            else:
                pos_x = right if is_left else left
                
            pos = (pos_x + noise1 - 3, center_y + noise2 - 3)

            zero_im = Image.fromarray(zero.numpy())
            scale_factor = 0.25
            new_size = (int(zero_im.size[0] * scale_factor), int(zero_im.size[1] * scale_factor))
            zero_im = zero_im.resize(new_size)
            treatment.paste(zero_im, pos)

            effective_treatment = np.random.random() < EFFICACY

            if effective_treatment and is_accurate:
                
                dark = Image.new('L', img.size, 0)
                still_sick = Image.blend(img, dark, 0.2)
                outcome = still_sick
            else:
                
                outcome = patient


            treatment.save(f'vbdata/causal_xray/{split}/xray_{i}_x.png')
            outcome.save(f'vbdata/causal_xray/{split}/xray_{i}_y.png')
            patient.save(f'vbdata/causal_xray/{split}/xray_{i}_z.png')


    def __len__(self):
        if self.split == 'train':
            return 78468
        else:
            return 22433


    def __getitem__(self, n):

        treatment = Image.open(f'vbdata/causal_xray/{self.split}/xray_{n}_x.png')
        outcome = Image.open(f'vbdata/causal_xray/{self.split}/xray_{n}_y.png')

        patient = Image.open(f'vbdata/causal_xray/{self.split}/xray_{n}_z.png')

        X = add_noise(transforms.ToTensor()(treatment)).flatten()
        
        Y = add_noise(transforms.ToTensor()(outcome)).flatten()

        Z = add_noise(transforms.ToTensor()(patient)).flatten()
        return X, Y, Z
    

class InterventionalXrayCausalDataset(Dataset):
    def __init__(self, accuracy, split='test', download=False):
        
        self.base_dataset = ChestMNIST(split=split, download=download)
        if download:
            self.create_data(split)

        mnist = torchvision.datasets.MNIST(root='../data', train=split, download=True)
        self.zeros = mnist.data[mnist.targets==0]

        self.split = split
        
        self.accuracy = accuracy

        np.random.seed(seed=0)

    def __getitem__(self, n):
        
        LEFT = 0.5
        ACCURACY = self.accuracy
        EFFICACY = 0.7

        img = self.base_dataset.__getitem__(n)[0]
        zero = self.zeros[np.random.randint(0, len(self.zeros))]

        is_left = np.random.random() < LEFT

        left = np.random.randint(low=8, high=11)
        right = np.random.randint(low=17, high=20)
        center_x = left if is_left else right


        center_y = np.random.randint(low=10, high=12)
        height = np.random.randint(low=3, high=4)
        width = np.random.randint(low=3, high=4)

        tumor = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(tumor)
        draw.ellipse((center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2), fill=255)
        patient = Image.blend(img, tumor, 0.2)

        draw = ImageDraw.Draw(tumor)
        draw.ellipse((center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2), fill=255)


        noise1 = np.random.randint(low=-1, high=1)
        noise2 = np.random.randint(low=-1, high=1)
        treatment = Image.new('L', img.size, 0)


        is_accurate = np.random.random() < ACCURACY
        if is_accurate:
            pos_x = left if is_left else right
        else:
            pos_x = right if is_left else left
            
        pos = (pos_x + noise1 - 3, center_y + noise2 - 3)

        zero_im = Image.fromarray(zero.numpy())
        scale_factor = 0.25
        new_size = (int(zero_im.size[0] * scale_factor), int(zero_im.size[1] * scale_factor))
        zero_im = zero_im.resize(new_size)
        treatment.paste(zero_im, pos)

        effective_treatment = np.random.random() < EFFICACY

        if effective_treatment and is_accurate:
            
            dark = Image.new('L', img.size, 0)
            still_sick = Image.blend(img, dark, 0.2)
            outcome = still_sick
        else:
            
            outcome = patient

        X = add_noise(transforms.ToTensor()(treatment)).flatten()
        
        Y = add_noise(transforms.ToTensor()(outcome)).flatten()

        Z = add_noise(transforms.ToTensor()(patient)).flatten()

        return X, Y
    
    def __len__(self):
        return len(self.base_dataset)

