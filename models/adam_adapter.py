import logging
import numpy as np
from scipy.linalg import inv, det
import torch
import pickle
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,MultiBranchCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy


# Tune the model at first session with adapter, and then classify with NCM/TEEN/FeCAM.
# To run TEEN, set fecam to false and calibartion to true
# To run NCM, set calibration and fecam to false

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["backbone_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')

        if 'resnet' in args['backbone_type']:
            self._network = SimpleCosineIncrementalNet(args, True)
            self. batch_size=128
            self.init_lr=args["init_lr"] if args["init_lr"] is not None else  0.01
        else:
            if self.args['resume']:
                self._network = SimpleVitNet(args, False)
            else:
                self._network = SimpleVitNet(args, True)
            self. batch_size= args["batch_size"]
            self. init_lr=args["init_lr"]
        
        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args
        self.Q = torch.zeros(768, self.args["nb_classes"])
        self.G = torch.zeros(768, 768)
        self.cov_mats, self.base_cov_mats = [], []
        self.ridge = 1
        self.beta = 0.5

    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self, trainloader, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list=np.unique(self.train_dataset.labels)
        proto_list = []

        for class_index in class_list:
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            proto_list.append(proto)
            
            if args['fecam']:
                cov = torch.cov(embedding.T)
                self.cov_mats.append(cov)
                if self._cur_task == 0:
                    self.base_cov_mats.append(cov)
                    
            if not args['calibration'] or self._cur_task == 0:
                self._network.fc.weight.data[class_index] = proto
        
        base_protos = self._network.fc.weight.data[:args['init_cls']].detach().cpu()
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        softmax_t = 16
        proto_list = torch.stack(proto_list).detach().cpu()
        cur_protos = F.normalize(proto_list, p=2, dim=-1)
        weights = torch.mm(cur_protos, base_protos.T) * softmax_t
        norm_weights = torch.softmax(weights, dim=1)
        
        if args['calibration'] and self._cur_task != 0:
            alpha = 0.9  # 0.9 for CUB200, Aircrafts, Cars and 0.75 for CIFAR100
            delta_protos = torch.matmul(norm_weights, base_protos)
            delta_protos = F.normalize(delta_protos, p=2, dim=-1)
            updated_protos = alpha * cur_protos + (1-alpha) * delta_protos
            for idd, class_index in enumerate(class_list):
                self._network.fc.weight.data[class_index] = updated_protos[idd]

        if args['fecam']:
            if not args['calibration'] or self._cur_task == 0:
                self.ridge = 100
                for idd, class_index in enumerate(class_list):
                    self.cov_mats[class_index] = torch.corrcoef(self.shrink_cov(self.cov_mats[class_index],self.ridge))
            else:
                beta = 1.0
                for idd, class_index in enumerate(class_list):
                    delta_covs = norm_weights[idd].view(args['init_cls'], 1, 1)*torch.stack(self.base_cov_mats[:args['init_cls']],0)
                    self.cov_mats[class_index] = beta*torch.sum(delta_covs,0) + beta*self.cov_mats[class_index]
                    self.cov_mats[class_index] = torch.corrcoef(self.shrink_cov(self.cov_mats[class_index],self.ridge))                  

        return model


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            self.shot = self.args["shot"]
        else:
            self.shot = None

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", shot=self.shot, )

        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", shot=self.shot, )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        
        self._network.to(self._device)
        
        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            if total_params != total_trainable_params:
                for name, param in self._network.named_parameters():
                    if param.requires_grad:
                        print(name, param.numel())
            if self.args['optimizer']=='sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
            elif self.args['optimizer']=='adam':
                optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
            if not self.args['resume']:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
                self.save_checkpoint("weights/{}_{}_{}_{}_{}".format(self.args["dataset"],self.args["model_name"],self.args["seed"],self.args["init_cls"],self.args["increment"]))
                self._network.to(self._device)
            else:
                self._network.load_state_dict(torch.load("weights/{}_{}_{}_{}_{}_{}.pkl".format(self.args["dataset"],self.args["model_name"],self.args["seed"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
                self._network.to(self._device)
            # self.construct_dual_branch_network()
        else:
            pass
        self.replace_fc(train_loader_for_protonet, self._network, self.args)
            

    def construct_dual_branch_network(self):
        network = MultiBranchCosineIncrementalNet(self.args, True)
        network.construct_dual_branch_network(self._network)
        self._network=network.to(self._device)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
