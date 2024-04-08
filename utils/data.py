import numpy as np
import torch
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels, list2dict, text_read
import os


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t


class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCUB200(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        path = "data/CUB_200/"
        self._pre_operate(path)

        self.train_data, self.train_targets = self.SelectData(self._train_data, self._train_targets)
        self.test_data, self.test_targets = self.SelectData(self._test_data, self._test_targets)

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'images.txt')
        split_file = os.path.join(root, 'train_test_split.txt')
        class_file = os.path.join(root, 'image_class_labels.txt')
        id2image = list2dict(text_read(image_file))
        id2train = list2dict(text_read(split_file))  # 1: train images; 0: test images
        id2class = list2dict(text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self._train_data, self._test_data = [], []
        self._train_targets, self._test_targets = [], []
        self.train_data2label, self.test_data2label = {}, {}
        for k in train_idx:
            image_path = os.path.join(root, 'images', id2image[k])
            self._train_data.append(image_path)
            self._train_targets.append(int(id2class[k]) - 1)
            self.train_data2label[image_path] = (int(id2class[k]) - 1)

        for k in test_idx:
            image_path = os.path.join(root, 'images', id2image[k])
            self._test_data.append(image_path)
            self._test_targets.append(int(id2class[k]) - 1)
            self.test_data2label[image_path] = (int(id2class[k]) - 1)

    def SelectData(self, data, targets):
        data_tmp = []
        targets_tmp = []
        for j in range(len(data)):
            data_tmp.append(data[j])
            targets_tmp.append(targets[j])

        return np.array(data_tmp), np.array(targets_tmp)
    
    
class iCARS(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()
    
    def download_data(self):
        train_dataset = datasets.StanfordCars(root="data/", split='train', download=True)
        test_dataset = datasets.StanfordCars(root="data/", split='test', download=True)

        data, targets = [], []
        for j in range(len(train_dataset)):
            data.append(np.array(transforms.Resize((224,224))(train_dataset[j][0])))
            targets.append(train_dataset[j][1])

        self.train_data = np.array(data)
        self.train_targets = np.array(targets)
        torch.save(self.train_data, "data/cars/train_data.pkl")
        torch.save(self.train_targets, "data/cars/train_targets.pkl")

        data, targets = [], []
        for j in range(len(test_dataset)):
            data.append(np.array(transforms.Resize((224,224))(test_dataset[j][0])))
            targets.append(test_dataset[j][1])

        self.test_data = np.array(data)
        self.test_targets = np.array(targets)
        torch.save(self.test_data, "data/cars/test_data.pkl")
        torch.save(self.test_targets, "data/cars/test_targets.pkl")

        # Generate pickle files for the first time and re-use the saved pickle files later
        # self.train_data = torch.load("data/cars/train_data.pkl")
        # self.train_targets = torch.load("data/cars/train_targets.pkl")
        # self.test_data = torch.load("data/cars/test_data.pkl")
        # self.test_targets = torch.load("data/cars/test_targets.pkl")

        print('Dataset done')


class iAircrafts(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()
    
    def download_data(self):
        train_dataset = datasets.FGVCAircraft(root="data/", split='trainval', download=True)
        test_dataset = datasets.FGVCAircraft(root="data/", split='test', download=True)

        data, targets = [], []
        for j in range(len(train_dataset)):
            data.append(np.array(transforms.Resize((224,224))(train_dataset[j][0])))
            targets.append(train_dataset[j][1])

        self.train_data = np.array(data)
        self.train_targets = np.array(targets)
        torch.save(self.train_data, "data/Air/train_data.pkl")
        torch.save(self.train_targets, "data/Air/train_targets.pkl")

        data, targets = [], []
        for j in range(len(test_dataset)):
            data.append(np.array(transforms.Resize((224,224))(test_dataset[j][0])))
            targets.append(test_dataset[j][1])

        self.test_data = np.array(data)
        self.test_targets = np.array(targets)
        torch.save(self.test_data, "data/Air/test_data.pkl")
        torch.save(self.test_targets, "data/Air/test_targets.pkl")

        # Generate pickle files for the first time and re-use the saved pickle files later
        # self.train_data = torch.load("data/Air/train_data.pkl")
        # self.train_targets = torch.load("data/Air/train_targets.pkl")
        # self.test_data = torch.load("data/Air/test_data.pkl")
        # self.test_targets = torch.load("data/Air/test_targets.pkl")

        print('Dataset done')
    
