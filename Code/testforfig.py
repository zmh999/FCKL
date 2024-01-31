import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import json

from sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, RobertaPAIRSentenceEncoder
from proto import Proto
import sys
import torch
from torch import optim, nn
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
# from process.utils import build_rel_vocab
# from process.processconcept import concept_net
# from process.utils import rel_vocab
# from process.processconcept import ConceptNet


class FewShotTestREFramework:

    def __init__(self, test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''

        self.test_data_loader = test_data_loader


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

    def test(self,
             model,
             B, N, K, Q,
             eval_iter,
             na_rate=0,
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
        print("")

        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
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
        #preds = []
        sne = [] ########################
        lists = []
        with torch.no_grad():
            for i in range(100):
                support, query, label = next(eval_dataset)
                # source, _, _ = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                        # source[k] = source[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()
                logits, pred, _ = model(support, query, N, K, N * Q, source=None, sne=sne)  ################Q * N + Q * na_rate
                # print(logits.shape)
                print(pred)
                newpred = pred.cpu()
                list = newpred.numpy().tolist()
                for i in list:
                    lists.append(i)
                # print(len(lists))
                # with open("pred-5-5-C.json", "w") as f: ####################################################
                #     f.write(str(lists))
        return lists


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=True, mask_entity=False):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('../pretrain/bert-base-uncased/vocab.txt')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        # listt = []
        # for nama, p in self.bert.named_parameters():
        #     listt.append(nama)
        # listt = listt[-2:]
        # # print(self.bert.state_dict()[listt[-1]])
        # for name in listt:
        #     torch.nn.init.normal_(self.bert.state_dict()[name], mean=0.0, std=1.0)
        # # print(self.bert.state_dict()[listt[-1]])
        # print('init finshed')
        if os.path.exists('stict.npy'):
            self.mlmdict = np.load('stict.npy', allow_pickle=True).item()
        else:
            self.mlmdict = {}

    def forward(self, inputs):
        if not self.cat_entity_rep:
            _, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
            return x
        else:
            outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            # CLS = outputs[1]
            # print(outputs[0].shape)
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            h2_state = outputs[0][tensor_range, inputs["pos2"]]
            t_state = outputs[0][tensor_range, inputs["pos3"]]
            t2_state = outputs[0][tensor_range, inputs["pos4"]]
            e1 = outputs[0][tensor_range, inputs["pos5"]]
            e2 = outputs[0][tensor_range, inputs["pos6"]]
            # R = outputs[0][tensor_range, inputs["pos8"]]
            rela = inputs["pos7"]

            # state = torch.cat((h_state + e1,t_state + e2, h2_state-t2_state), -1)  #U9  93.74
            state = torch.cat((h_state + e1, t_state + e2, h2_state - t2_state), -1)  #
            # state = torch.cat((h_state, t_state, h2_state - t2_state), -1)  # -EME

            # state = torch.cat((h_state, h2_state, t_state, t2_state), -1)
            # state = h_state+h2_state+t_state+t2_state
            # state = h_state+h2_state-t_state-t2_state
            # state = torch.cat((h_state+h2_state, t_state+t2_state), -1)
            return state, outputs[0], rela, outputs[1]
            # return  outputs[1]

    def tokenize(self, raw_tokens, pos_head, pos_tail, Ehead, Etail, istrain=True):
        # token -> index

        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        pos3_in_index = 1
        pos4_in_index = 1
        pos5_in_index = 1
        pos6_in_index = 1
        pos7_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (
                    pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                # if not istrain:
                #     tokens+=self.tokenizer.tokenize(e1maskword) ##############
                # tokens += self.tokenizer.tokenize(rela)  ############## MLM2
                # tokens += Eheadtoken  ##############
                tokens.append('[unused2]')
                pos3_in_index = len(tokens)
                # tokens += Eheadtoken  ##############
            if cur_pos == pos_tail[-1]:
                # if not istrain:
                #     tokens += self.tokenizer.tokenize(e2maskword) #######
                # tokens += self.tokenizer.tokenize(rela)  ##############MLM2
                # tokens += Etailtoken  ##############
                tokens.append('[unused3]')
                pos4_in_index = len(tokens)
                # tokens += Etailtoken  ##############
            cur_pos += 1

        pos7_in_index = len(tokens)
        # print(tokens)
        # tokens[-1] = ','
        # print(tokens)
        # print('')
        #####
        # tokens += relatoken #######
        ####
        ####
        E1 = ['\"'] + [raw_tokens[t] for t in pos_head] + ['\"']
        E2 = ['\"'] + [raw_tokens[t] for t in pos_tail] + ['\"']
        zhishi1 = 'means'  # indicate    :   means :
        zhishi2 = 'and'
        # tokens += self.tokenizer.tokenize(' in this sentence : ')
        # tokens += self.tokenizer.tokenize('the public domain or medical domain entity')
        for token in E1:
            tokens += self.tokenizer.tokenize(token)
        tokens += self.tokenizer.tokenize(zhishi1)
        # tokens += Eheadtoken  ##############
        tokens.append('[unused5]')
        pos5_in_index = len(tokens)
        # tokens += self.tokenizer.tokenize('in public domain or medical domain')
        # tokens += Eheadtoken  ##############  放到上边？  #################
        # tokens+=self.tokenizer.tokenize(zhishi2)
        # tokens += self.tokenizer.tokenize('to')
        # for token in E2:
        #     tokens += self.tokenizer.tokenize(token)
        tokens += self.tokenizer.tokenize(',')
        # tokens += relatoken  ###############
        # tokens += self.tokenizer.tokenize(',')
        # tokens += self.tokenizer.tokenize('the public domain or medical domain entity')
        for token in E2:
            tokens += self.tokenizer.tokenize(token)
        tokens += self.tokenizer.tokenize(zhishi1)
        # tokens += Eheadtoken  ##############
        tokens.append('[unused6]')
        pos6_in_index = len(tokens)
        # tokens += self.tokenizer.tokenize('in public domain or medical domain')
        # tokens += self.tokenizer.tokenize('.')
        # tokens += self.tokenizer.tokenize('to')
        # for token in E1:
        #     tokens += self.tokenizer.tokenize(token)
        # tokens += Etailtoken  ########
        # print(tokens)

        # pos8_in_index = (tokens.index('[unused8]'))+1
        pos8_in_index = 1
        #####

        # print(tokens)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
        pos3_in_index = min(self.max_length, pos3_in_index)
        pos4_in_index = min(self.max_length, pos4_in_index)
        pos5_in_index = min(self.max_length, pos5_in_index)
        pos6_in_index = min(self.max_length, pos6_in_index)
        pos7_in_index = min(self.max_length, pos7_in_index)
        pos8_in_index = min(self.max_length, pos8_in_index)

        ansid = [pos1_in_index - 1, pos3_in_index - 1, pos2_in_index - 1, pos4_in_index - 1]
        posid = sorted([pos4_in_index - 1, pos3_in_index - 1, pos2_in_index - 1, pos1_in_index - 1], reverse=True)
        finalid = 0
        for oneid in posid:
            if '[unused' in tokens[oneid]:
                finalid = oneid
                break
        for id in range(4):
            if not '[unused' in tokens[ansid[id]]:
                ansid[id] = finalid
                # ansid[id] = 0

        # print(tokens)
        # print(tokens[pos1_in_index-1], tokens[pos3_in_index-1],tokens[pos2_in_index-1], tokens[pos4_in_index-1])
        # print(tokens[0])
        return indexed_tokens, ansid[0], ansid[1], ansid[2], ansid[3], pos5_in_index - 1, pos6_in_index - 1, \
               pos7_in_index - 1, pos8_in_index - 1, mask


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root, single=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.istrain = True
        self.single = single
        # if 'train' in path:
        #     self.istrain = True
        # else:
        #     self.istrain = False

    def __getraw__(self, item):
        word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask = self.encoder.tokenize(item['tokens'],
                                                                                           item['h'][2][0],
                                                                                           item['t'][2][0],
                                                                                           item['h'][0],
                                                                                           item['t'][0],
                                                                                           self.istrain)
        return word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask

    def __additem__(self, d, word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['pos3'].append(pos3)
        d['pos4'].append(pos4)
        d['pos5'].append(pos5)
        d['pos6'].append(pos6)
        d['pos7'].append(pos7)
        d['pos8'].append(pos8)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)

        target_classes = self.classes[5:]
        # target_classes = self.classes[0:5]
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'pos3': [], 'pos4': [], 'pos5': [], 'pos6': [], 'pos7': [],
                       'pos8': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'pos3': [], 'pos4': [], 'pos5': [], 'pos6': [], 'pos7': [],
                     'pos8': [], 'mask': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))
        if self.single:
            qnum = random.sample(list(range(self.N)), 1)[0]
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask = self.__getraw__(
                    self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                pos3 = torch.tensor(pos3).long()
                pos4 = torch.tensor(pos4).long()
                pos5 = torch.tensor(pos5).long()
                pos6 = torch.tensor(pos6).long()
                pos7 = torch.tensor(pos7).long()
                pos8 = torch.tensor(pos8).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask)
                else:
                    if not self.single:
                        self.__additem__(query_set, word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask)
                    elif qnum == i:
                        self.__additem__(query_set, word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask)

                count += 1
            if not self.single:
                query_label += [i] * self.Q
            elif qnum == i:
                query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.json_data[cur_class]))),
                1, False)[0]
            word, pos1, pos2, pos3, pos4, mask = self.__getraw__(
                self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label

    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'pos3': [], 'pos4': [], 'pos5': [], 'pos6': [], 'pos7': [],
                     'pos8': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'pos3': [], 'pos4': [], 'pos5': [], 'pos6': [], 'pos7': [],
                   'pos8': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label


def get_loader(name, encoder, N, K, Q, batch_size, single=False,
               num_workers=0, collate_fn=collate_fn, na_rate=0, root='../data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root, single)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='val_pubmed', #############################
                        help='train file')

    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--K', default=1, type=int,######################################
                        help='K shot')
    parser.add_argument('--Q', default=1, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--model', default='proto',
                        help='model name')
    parser.add_argument('--encoder', default='bert',
                        help='encoder: cnn or bert or roberta')
    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--load_ckpt', default='checkpoint/51/proto-bert-train_wiki-val_pubmed-5-1-adv_pubmed_unsupervised-catentity-51W3D.pth.tar',
                        help='load ckpt')

    parser.add_argument('--ckpt_name', type=str, default='',
                        help='checkpoint name.')

    opt = parser.parse_args()

    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    encoder_name = opt.encoder
    max_length = opt.max_length
    pretrain_ckpt = 'bert-base-uncased'
    sentence_encoder = BERTSentenceEncoder(
        pretrain_ckpt,
        max_length)
    test_data_loader = get_loader(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, batch_size=batch_size)
    model = Proto(sentence_encoder)
    if torch.cuda.is_available():
        model.cuda()
    ckpt = opt.load_ckpt
    framework = FewShotTestREFramework(test_data_loader)
    result = framework.test(model, batch_size, N, K, Q, 1, 0, ckpt=ckpt)
    print(len(result))


if __name__ == "__main__":
    main()