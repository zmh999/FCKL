import os
import sklearn.metrics
import numpy as np
import sys
import time
import sentence_encoder
import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import math
import matplotlib.pyplot as plt
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(my_sentence_encoder)
        self.cost = nn.CrossEntropyLoss()
        self.cost2 = nn.NLLLoss()
    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        # logitssft = torch.softmax(logits,dim=-1).view(-1,N)
        # logits = torch.log(logitssft)
        # return self.cost2(1/logitssft*logits.view(-1, N), label.view(-1))
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv
        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.d = d
            self.d.cuda()

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              na_rate=0,
              learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              bert_optim=False,
              warmup=True,
              warmup_step=1000,
              grad_iter=1,
              fp16=False,
              pair=False,
              adv_dis_lr=1e-1,
              adv_enc_lr=1e-1,
              use_sgd_for_bert=False):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        print("Start training...")
    
        # Init
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            if use_sgd_for_bert:
                optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
            else:
                optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
            if self.adv:
                optimizer_encoder = AdamW(parameters_to_optimize, lr=1e-5, correct_bias=False)
                # scheduler_encoder = get_linear_schedule_with_warmup(optimizer_encoder, num_warmup_steps=warmup_step,num_training_steps=train_iter)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter)
            # scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
            #                                             num_training_steps=train_iter)
        else:
            optimizer = pytorch_optim(model.parameters(),
                    learning_rate, weight_decay=weight_decay)
            if self.adv:
                optimizer_encoder = pytorch_optim(model.parameters(), lr=adv_enc_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if self.adv:
            optimizer_dis = pytorch_optim(self.d.parameters(), lr=adv_dis_lr)
            # scheduler_dis = get_linear_schedule_with_warmup(optimizer_dis, num_warmup_steps=warmup_step,
            #                                             num_training_steps=train_iter)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()
        if self.adv:
            self.d.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        DK_gen = 0.1
        DK_dis = 1
        PS = []
        PST = []
        for it in range(start_iter, start_iter + train_iter):
            # print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',it)
            if pair:
                batch, label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in batch:
                        batch[k] = batch[k].cuda()
                    label = label.cuda()
                logits, pred = model(batch, N_for_train, K, 
                        Q * N_for_train + na_rate * Q)
            else:
                support, query, label = next(self.train_data_loader)
                # target,_,_ = next(self.val_data_loader)
                target = next(self.adv_data_loader)
                # source = target
                # for k in support:
                #         source[k] = support[k][N_for_train*B:]
                #         source[k] = source[k].cuda()
                #         support[k] = support[k][0:N_for_train*B]
                #         query[k] = query[k][0:N_for_train*B]
                # label = torch.cat((label[0:N_for_train],label[0:N_for_train]),dim=-1)

                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()
                    for k in target:
                        target[k] = target[k].cuda()
                        # target[k] = target[k][0].unsqueeze(0)
                logits, pred, adv_data = model(support, query,
                        N_for_train, K, Q * N_for_train + na_rate * Q, label=label, target=target, MYlist=[PS,PST])
            loss = model.loss(logits, label) / float(grad_iter)
            loss = loss+adv_data[5]
            right = model.accuracy(pred, label)
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10)
            else:
                # loss = dbloss
                loss.backward()
                # loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            
            if it % grad_iter == 0:
                optimizer.step()
                # if (it+1) <= 7000:
                #     scheduler.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Adv part
            la = 1
            if self.adv and False:
                # print('adv')
                # support_adv = next(self.adv_data_loader)
                # if torch.cuda.is_available():
                #     for k in support_adv:
                #         support_adv[k] = support_adv[k].cuda()




                #3
                # features_ori = model.sentence_encoder(support) ###########
                # features_adv = model.sentence_encoder(support_adv)
                features_ori = adv_data[0]  ###########
                features_adv = adv_data[1]
                # print(features_ori.shape)
                # print(features_adv.shape)

                # features_ori = features_ori - torch.mean(features_ori, dim=0)
                # features_adv = features_adv - torch.mean(features_adv, dim=0)
                features = torch.cat([features_ori, features_adv], 0).unsqueeze(1)
                total = features.size(0)
                dis_labels = torch.cat([torch.zeros((total // 2)).long().cuda(),
                    torch.ones((total // 2)).long().cuda()], 0)
                dis_logits = self.d(features)
                loss_dis = self.adv_cost(dis_logits, dis_labels) * 1
                _, pred = dis_logits.max(-1)
                right_dis = float((pred == dis_labels).long().sum()) / float(total)

                loss_encoder = self.adv_cost(dis_logits, 1 - dis_labels) * 0.1

                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()
                for p in model.parameters():
                    p.requires_grad = False
                loss_dis.backward(retain_graph=True)
                for p in model.parameters():
                    p.requires_grad = True

                for p in self.d.parameters():
                    p.requires_grad = False
                loss_encoder.backward()
                for p in self.d.parameters():
                    p.requires_grad = True

                optimizer_dis.step()
                optimizer_encoder.step()
                # scheduler_dis.step()
                #
                iter_loss_dis += self.item(loss_dis.data)
                iter_right_dis += right_dis
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            # print(optimizer_encoder.state_dict()['param_groups'][0]['lr'])
            # print(optimizer_dis.state_dict()['param_groups'][0]['lr'])
            # print('')
            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            if self.adv:
                if (it + 1) % 100 == 0:
                    print('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}'
                        .format(it + 1, iter_loss / iter_sample,
                            100 * iter_right / iter_sample,
                            iter_loss_dis / iter_sample,
                            100 * iter_right_dis / iter_sample) + '\r')
            else:
                if (it + 1) % 100 == 0:
                    print('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')

            if (it + 1) % val_step == 0:
                # print(model.Scen)
                # print(model.Tcen)
                # print(model.state_dict()['a1'],model.state_dict()['a2'],model.state_dict()['a3'],model.state_dict()['a4'],model.state_dict()['a5'])
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter, 
                        na_rate=na_rate, pair=pair)
                print("EVAL RESULT: %.2f" % (acc * 100))
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                DK_gen = 0.1
                DK_dis = math.sqrt(1-(iter_right_dis / iter_sample)**2)
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.
                print('')
        # np.save('stict.npy', model.sentence_encoder.mlmdict)#########

        print("\n####################\n")
        print("THE BEST EVAL RESULT: %.2f" % (best_acc * 100))
        print("Finish training " + model_name)
        # PSarray = np.array(PS)
        # PSTarray = np.array(PST)
        # XX = np.linspace(0,10000,10000)
        # plt.xlabel('Number of iterations')
        # plt.ylabel('Correlation coefficient')
        # plt.plot(XX, PSarray, label="$PS$", color="red", linewidth=1)
        # plt.plot(XX, PSTarray, label="$PST$", color="blue", linewidth=1)
        # plt.legend()
        # plt.show()
        # plt.savefig(model_name)
        # # print(PSarray)
        # # print(PSTarray)
        # np.save('NPSarray.npy', PSarray)
        # np.save('NPSTarray.npy', PSTarray)

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            na_rate=0,
            pair=False,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''

        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.test_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            for it in range(eval_iter):
                # print('vvvvvvvvvvvvvvvvvvvvvvv',it)
                if pair:
                    batch, label = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in batch:
                            batch[k] = batch[k].cuda()
                        label = label.cuda()
                    logits, pred = model(batch, N, K, Q * N + Q * na_rate)
                else:
                    support, query, label = next(eval_dataset)
                    # source, _, _ = next(self.train_data_loader)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                            # source[k] = source[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        label = label.cuda()
                    logits, pred,_ = model(support, query, N, K, N*Q, source=None)################Q * N + Q * na_rate

                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_sample += 1
                # if (it+1)%100==0:
                #     print('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
        return iter_right / iter_sample
