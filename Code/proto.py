import random
import sys
sys.path.append('..')
import torch
from torch import autograd, optim, nn
import framework
from torch.autograd import Variable
from torch.nn import functional as F
import math
from sklearn import metrics
import numpy as np

class Proto(framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dot=False):
        framework.FewShotREModel.__init__(self, sentence_encoder)
        self.drop = nn.Dropout(0) #########################Drop
        self.cost = nn.CrossEntropyLoss()
        self.dimdrop1d = nn.Dropout(0.1)
        # self.ww1 = nn.Parameter(torch.Tensor([1]))
        # self.ww2 = nn.Parameter(torch.Tensor([1]))
        kernel = 5
        p = 2
        NS = 1 * 5 * 1 ################### qq
        NQ = 1 * 5 * 1
        NT = 1 * 5 * 1
        shots = 1

        self.convS = nn.Sequential(  # small
            nn.Conv1d(NS+1+NT, 256, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 128, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 1, kernel, padding=p), ##################
            nn.LeakyReLU(),
        )
        # self.convSup = nn.Sequential(  # small
        #     nn.Conv1d(NS, 256, kernel, padding=p),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.1),
        #     nn.Conv1d(256, 128, kernel, padding=p),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.1),
        #     nn.Conv1d(128, 64, kernel, padding=p),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.1),
        #     nn.Conv1d(64, 1, kernel, padding=p),  ##################
        #     nn.LeakyReLU(),
        # )
        self.convT = nn.Sequential(  # small
            nn.Conv1d(NT, 256, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 128, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 1, kernel, padding=p),  ##################
            nn.LeakyReLU(),
        )


        self.convA = nn.Sequential(  # small
            nn.Conv1d(1, 256, kernel, padding=p),  # G
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 128, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 1, kernel, padding=p),
            # nn.LeakyReLU(),
        )
        self.convB = nn.Sequential(  # small
            nn.Conv1d(1, 256, kernel, padding=p),  # G
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 128, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 1, kernel, padding=p),
            # nn.LeakyReLU(),
        )
        self.convC = nn.Sequential(  # small
            nn.Conv1d(1, 256, kernel, padding=p),  # G
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 128, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, kernel, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 1, kernel, padding=p),
            # nn.LeakyReLU(),
        )


    def getloss(self, logits, label):
        if label is None:
            return 0
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def __dist__(self, x, y, dim):
            simcos = torch.cosine_similarity(y, x, dim)
            # simdis = -(torch.pow(x - y, 2)).sum(dim)
            #
            # dismean = torch.mean(simdis, dim=2, keepdim=True)  # cos dis各自标准化
            # disstd = torch.std(simdis, dim=2, keepdim=True)
            # simdis = (simdis - dismean) / disstd

            cosmean = torch.mean(simcos, dim=2, keepdim=True)
            cosstd = torch.std(simcos, dim=2, keepdim=True)
            simcos = (simcos - cosmean) / cosstd
            #


            # return simcos + simdis
            #
            return simcos
            # return simdis



    def __batch_dist__(self, S, Q, f=None):
        # if f is None:
        #     return torch.einsum('iqd,ijd->iqj', Q, S)
        # dis =  torch.einsum('iqd,ijd->iqjd', Q, S)
        # return dis
        if f is None:
            return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)  #[B N D]  [B totalQ D]
        else:
            return torch.cosine_similarity(Q.unsqueeze(2), S.unsqueeze(1), 3)
            # return -(torch.pow(S.unsqueeze(1) - Q.unsqueeze(2), 2)).sum(3)
            # return ((Q.unsqueeze(2))*(S.unsqueeze(1))).sum(3)

    def forward(self, support, query, N, K, total_Q, label=None, target=None, source=None, MYlist=None, sne=None):
        adv_S = None
        adv_T = None
        corloss = 0

        if not target is None:
            support_emb, Sout, Srela, SCLS = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
            query_emb, Qout, Qrela, QCLS = self.sentence_encoder(query)  # (B * total_Q, D)
            target_emb, Tout, Trela, TCLS = self.sentence_encoder(target)  # [10, D]

            # Drop
            hidden_size = support_emb.size(-1)
            support = self.drop(support_emb)
            query = self.drop(query_emb)
            target = self.drop(target_emb)


            # 减均值

            mean_sup = torch.mean(support, dim=0)
            mean_que = torch.mean(query, dim=0)
            mean_tar = torch.mean(target, dim=0)
            support =support - mean_sup
            query = query - mean_sup
            target = target - mean_tar

            adv_S = torch.mean(support.view(N, K, hidden_size), dim=1)
            adv_T = target


            # conv
            support = support.view(-1, hidden_size)  # (B, N, K, D)
            query = query.view(-1, hidden_size)
            # source = torch.cat((support, query), dim=0)
            target = target.view(-1, hidden_size)
            # query = query.view(-1, hidden_size)
            # 分开卷积
            # featsu = support.unsqueeze(0)  ############################################################################
            # featque = query.unsqueeze(0)
            feattar = target.unsqueeze(0)
            # featall = torch.cat((support, target), dim=0).unsqueeze(0)
            # featall = self.convST(featall).squeeze(0)

            # featsup = self.convS(featsup).squeeze(0)
            feattar = self.convT(feattar).squeeze(0)
            # featsu = self.convSup(featsu).squeeze(0)
            featsuplist = []
            for i,que in enumerate(query):
                source = torch.cat((support,que.unsqueeze(0),target), dim=0)
                featsup = self.convS(source.unsqueeze(0)).squeeze(0)
                featsuplist.append(featsup.squeeze(0))
            # for i,featlist in enumerate(featsuplist):
            #     query[i] +=  (query[i] * featlist).squeeze(0)
            # print(query)
            featsuplist = torch.stack(featsuplist, dim=0)
            # feat_t = (self.a1*featsup+self.a2*feattar)/2

            # 2dconv
            featsup2d = support.view(-1, 1, hidden_size)
            feattar2d = target.view(-1, 1, hidden_size)
            featque2d = query.view(-1, 1 ,hidden_size)
            featsup2d = self.convA(featsup2d)
            feattar2d = self.convB(feattar2d)
            featque2d = self.convC(featque2d)
            # print(featsup2d.shape)

            # featsup2d = featsup2d * feattar
            # feattar2d = feattar2d * featsup
            featsup2d = featsup2d.view(-1, hidden_size)
            feattar2d = feattar2d.view(-1, hidden_size)
            featque2d = featque2d.view(-1, hidden_size)


            support = support + featsup2d
            query = query + featque2d + featsuplist
            target = target + feattar2d   # T





            # adv_supf = support.squeeze(0)
            # adv_tarf = target.squeeze(0)

            # SQT Out
            span = 3
            Semb = support.view(-1, span, 768)
            Qemb = query.view(-1, span, 768)
            Temb = target.view(-1, span, 768)

            # feaS = torch.sum(featsu.view(-1, 768), dim=0)
            feaT = torch.sum(feattar.view(-1, 768), dim=0)
            # feaT = 1

            k_of_T = 1
            numcountSR = torch.Tensor([[0] * 128] * N * K).cuda()
            numcountQR = torch.Tensor([[0] * 128] * N).cuda()
            numcountTR = torch.Tensor([[0] * 128] * N * k_of_T).cuda()
            numcountSRS = torch.Tensor([[0] * 128] * N * K).cuda()
            numcountQRS = torch.Tensor([[0] * 128] * N).cuda()
            numcountTRS = torch.Tensor([[0] * 128] * N * k_of_T).cuda()
            tensor_rangeSR = torch.arange(N * K)
            tensor_rangeQR = torch.arange(N)
            tensor_rangeTR = torch.arange(N * k_of_T)
            # print(Sout.shape)
            for i in range(Sout.shape[0]):
                sentence = Sout[i]  # [128 768]
                senemb = Semb[i]  # [3 768]
                # one_emb = SCLS[i]+senemb[2]
                # W_sub = (senemb[2]*(senemb[0]+senemb[1])).sum(0)
                # W_cls = (SCLS[i]*(senemb[0]+senemb[1])).sum(0)
                # W_S_C = torch.softmax(torch.Tensor([W_sub,W_cls]), dim=0)
                # one_emb = W_S_C[1]*SCLS[i] + W_S_C[0]*senemb[2]
                for j, word in enumerate(sentence[1:Srela[i]]):
                    # one_emb = (senemb[0] + senemb[1])  ##T2
                    one_emb = senemb[2]  ### T1
                    dianji = (one_emb * word*feaT ).sum(0)
                    # dianji = torch.cosine_similarity(one_emb, word, dim=0)
                    # one_emb = torch.cat((senemb, word.unsqueeze(0)), dim=0)
                    # dianji = self.Rconv(one_emb.unsqueeze(0))
                    # dianji = (dianji.sum(-1))[0].item()
                    numcountSR[i][j] = dianji
                    dianji = (feaT * word).sum(0)
                    numcountSRS[i][j] = dianji
            # _, SRid = torch.max(numcountSR, dim=1)
            # 最大的三个

            # SRidS = torch.sort(numcountSRS, descending=True, dim=1)[1]
            # SRidS = SRidS[:, 0:2].transpose(0, 1)  # [3 25]

            SRid = torch.sort(numcountSR, descending=True,dim=1)[1]
            SRid = SRid[:,0:2].transpose(0,1)  # [3 25]
            # WS = F.relu(torch.softmax(numcountSR*10,dim=1).unsqueeze(-1)-yuzhi)
            # SRres = (WS * Sout).sum(1)
            SRres = (Sout[tensor_rangeSR, SRid[0]] + Sout[tensor_rangeSR, SRid[1]])/2
            # SRres = (Sout[tensor_rangeSR, SRid[0]] + Sout[tensor_rangeSR, SRidS[0]]) / 2
            # SRres = torch.cat((Sout[tensor_rangeSR, SRid[0]] , Sout[tensor_rangeSR, SRidS[0]]),dim=1)

            # print(Sout[tensor_rangeSR, SRid[0]],Sout[tensor_rangeSR, SRid[1]],Semb)
            # print('')

            for i in range(Qout.shape[0]):
                sentence = Qout[i]  # [128 768]
                senemb = Qemb[i]  # [3 768]
                # one_emb = QCLS[i]+senemb[2]
                # W_sub = (senemb[2] * (senemb[0] + senemb[1])).sum(0)
                # W_cls = (QCLS[i] * (senemb[0] + senemb[1])).sum(0)
                # W_S_C = torch.softmax(torch.Tensor([W_sub, W_cls]), dim=0)
                # one_emb = W_S_C[1] * QCLS[i] + W_S_C[0] * senemb[2]

                # feaT_Q = torch.sum(featsuplist[i].view(-1, 768), dim=0)
                for j, word in enumerate(sentence[1:Qrela[i]]):
                    # one_emb = (senemb[0] + senemb[1])   ##T2
                    one_emb = senemb[2]
                    dianji = (one_emb * word*feaT  ).sum(0)
                    # dianji = torch.cosine_similarity(one_emb, word, dim=0)
                    # one_emb = torch.cat((senemb, word.unsqueeze(0)), dim=0)
                    # dianji = self.Rconv(one_emb.unsqueeze(0))
                    # dianji = (dianji.sum(-1))[0].item()
                    numcountQR[i][j] = dianji
                    dianji = (feaT * word).sum(0)
                    numcountQRS[i][j] = dianji
            # _, QRid = torch.max(numcountQR, dim=1)
            # 最大的三个

            # QRidS = torch.sort(numcountQRS, descending=True, dim=1)[1]
            # QRidS = QRidS[:, 0:2].transpose(0, 1)  # [3 25]

            QRid = torch.sort(numcountQR, descending=True, dim=1)[1]
            QRid = QRid[:, 0:2].transpose(0, 1)  # [3 25]
            # WQ = F.relu(torch.softmax(numcountQR, dim=1).unsqueeze(-1)-yuzhi)
            # QRres = (WQ * Qout).sum(1)
            QRres = (Qout[tensor_rangeQR, QRid[0]] + Qout[tensor_rangeQR, QRid[1]])/2
            # QRres = (Qout[tensor_rangeQR, QRid[0]] + Qout[tensor_rangeQR, QRidS[0]]) / 2
            # QRres = torch.cat((Qout[tensor_rangeQR, QRid[0]] , Qout[tensor_rangeQR, QRidS[0]]),dim=1)

            for i in range(Tout.shape[0]):
                sentence = Tout[i]  # [128 768]
                senemb = Temb[i]  # [3 768]
                # one_emb = TCLS[i]+senemb[2]
                # W_sub = (senemb[2] * (senemb[0] + senemb[1])).sum(0)
                # W_cls = (TCLS[i] * (senemb[0] + senemb[1])).sum(0)
                # W_S_C = torch.softmax(torch.Tensor([W_sub, W_cls]), dim=0)
                # one_emb = W_S_C[1] * TCLS[i] + W_S_C[0] * senemb[2]
                for j, word in enumerate(sentence[1:Trela[i]]):
                    # one_emb = (senemb[0] + senemb[1])  ##T2
                    one_emb = senemb[2]
                    dianji = (one_emb * word *feaT   ).sum(0)
                    # dianji = torch.cosine_similarity(one_emb, word, dim=0)
                    # one_emb = torch.cat((senemb, word.unsqueeze(0)), dim=0)
                    # dianji = self.Rconv(one_emb.unsqueeze(0))
                    # dianji = (dianji.sum(-1))[0].item()
                    numcountTR[i][j] = dianji
                    dianji = (feaT * word ).sum(0)
                    numcountTRS[i][j] = dianji
            # _, TRid = torch.max(numcountTR, dim=1)
            # 最大的三个
            # TRidS = torch.sort(numcountTRS, descending=True, dim=1)[1]
            # TRidS = TRidS[:, 0:2].transpose(0, 1)  # [3 25]

            TRid = torch.sort(numcountTR, descending=True, dim=1)[1]
            TRid = TRid[:, 0:2].transpose(0, 1)  # [3 25]
            # WT = F.relu(torch.softmax(numcountTR, dim=1).unsqueeze(-1)-yuzhi)
            # TRres = (WT * Tout).sum(1)
            TRres = (Tout[tensor_rangeTR, TRid[0]] + Tout[tensor_rangeTR, TRid[1]]) / 2
            # TRres = (Tout[tensor_rangeTR, TRid[0]]+Tout[tensor_rangeTR, TRidS[0]] )/2
            # TRres = torch.cat((Tout[tensor_rangeTR, TRid[0]] , Tout[tensor_rangeTR, TRidS[0]]),dim=1)


            # 最大的一个的附近的三个
            # TRres = (Tout[tensor_rangeTR, TRid-1] + Tout[tensor_rangeTR, TRid] + Tout[tensor_rangeTR, TRid+1])/3
            # TRres = Tout[tensor_rangeTR, TRid]

            support = support.view(-1, hidden_size)
            query = query.view(-1, hidden_size)
            target = target.view(-1, hidden_size)
            # print(support.shape)
            # print(SRres.shape)
            support = torch.cat((support, SRres), dim=1)
            query = torch.cat((query, QRres), dim=1)
            target = torch.cat((target, TRres), dim=1)
            #
            # support[:,-768:] = SRres
            # query[:,-768:] = QRres
            # target[:,-768:] =TRres

            hidden_size = support.size(-1)



            # # cos
            # support = support.view(-1, hidden_size)  ###################################################################
            # query = query.view(-1, hidden_size)
            # target = target.view(-1, hidden_size)
            # # sourceCor = torch.cat((support, query), dim=0)
            # BcN = 1 * (N * K)  #######################################  S
            # cos1 = self.__batch_dist__(support.unsqueeze(0), support.unsqueeze(0), -1)[0].squeeze(0)
            # cos2 = self.__batch_dist__(target.unsqueeze(0), support.unsqueeze(0), -1)[0].squeeze(0)
            # cosS = (cos1.sum() - BcN) / 2 / (BcN * (BcN - 1) / 2)
            # cosST = (cos2.sum()) / BcN / N
            #
            # cosloss1 = torch.abs(cosS - cosST)  # X
            #
            # corloss = corloss + cosloss1*1.6  # 93.91

            # corB
            support = support.view(-1, hidden_size) ###################################################################
            query = query.view(-1, hidden_size)
            target = target.view(-1, hidden_size)
            # sourceCor = torch.cat((support, query), dim=0)
            corsupport = torch.mean(support.view(N,K,hidden_size), dim=1)
            BcN = 1 * (N * 1)  #######################################  S
            cor = torch.corrcoef(torch.cat((corsupport, target), dim=0))
            corS = (cor[0:BcN, 0:BcN].sum() - BcN) / 2 / (BcN * (BcN - 1) / 2)
            corST = (cor[0:BcN, BcN:].sum()) / BcN / N
            corT = (cor[BcN:, BcN:].sum() - N) / 2 / (N * (N - 1) / 2)

            corloss1 = torch.abs(corS - corST)  # X
            corloss2 = torch.abs(corS - corT)  # Y
            corloss3 = torch.abs(corT - corST)  # Z
            # corloss =  torch.abs(corS)+torch.abs(corST )
            # print(corS, corST, corT)
            # MYlist[0].append(corS.data.item())
            # MYlist[1].append(corST.data.item())
            # print(corS)
            corloss = corloss + corloss1 * 1.5  # 93.91


            support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
            query = query.view(-1, total_Q, hidden_size)  # (B, total_Q, D)
            # target = target.view(-1, N, 1, hidden_size)
        else:
            support_emb, Sout, Srela, SCLS = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
            query_emb, Qout, Qrela, QCLS = self.sentence_encoder(query)  # (B * total_Q, D)
            # Drop
            hidden_size = support_emb.size(-1)
            support = self.drop(support_emb)
            query = self.drop(query_emb)

            # 减均值
            mean_sup = torch.mean(support, dim=0)
            mean_que = torch.mean(query, dim=0)
            support = support - mean_sup
            query = query - mean_sup
            #


            featsup2d = support.view(-1, 1, hidden_size)
            featque2d = query.view(-1, 1, hidden_size)
            featsup2d = self.convB(featsup2d)
            featque2d = self.convB(featque2d)
            #
            featsup2d = featsup2d.view(-1, hidden_size)
            featque2d = featque2d.view(-1, hidden_size)

            # #dim
            # sup2d = torch.mean(featsup2d.view(N, K, hidden_size), dim=1)
            # sup2d = torch.relu(torch.sign(sup2d))
            # # que2d = torch.relu(torch.sign(featque2d))
            #
            # featque2d = featque2d * torch.sign(torch.sum(sup2d, dim=0))

            support = support + featsup2d
            query = query + featque2d  #conv4


            # SQT Out
            span = 3
            Semb = support.view(-1, span, 768)
            Qemb = query.view(-1, span, 768)

            # feaS = torch.sum(featsup.view(-1, 768), dim=0)
            # feaT = torch.sum(feattar.view(-1, 768), dim=0)
            # feaT = featsup.squeeze(0)
            # feaT = feaT[-768:]
            # NforQinT = 1  # N      ___________________________________________________________test
            NforQinT = N  # N
            numcountSR = torch.Tensor([[0] * 128] * N * K).cuda()
            numcountQR = torch.Tensor([[0] * 128] * NforQinT).cuda()
            tensor_rangeSR = torch.arange(N * K)
            tensor_rangeQR = torch.arange(NforQinT)
            # print(Sout.shape)
            for i in range(Sout.shape[0]):
                sentence = Sout[i]  # [128 768]
                senemb = Semb[i]  # [3 768]
                # one_emb = SCLS[i]+senemb[2]
                # W_sub = (senemb[2] * (senemb[0] + senemb[1])).sum(0)
                # W_cls = (SCLS[i] * (senemb[0] + senemb[1])).sum(0)
                # W_S_C = torch.softmax(torch.Tensor([W_sub, W_cls]), dim=0)
                # one_emb = W_S_C[1] * SCLS[i] + W_S_C[0] * senemb[2]
                for j, word in enumerate(sentence[1:Srela[i]]):
                        # one_emb = (senemb[0] + senemb[1])   ##T2
                        one_emb = senemb[2]
                        dianji = (one_emb * word ).sum(0)
                        # dianji = torch.cosine_similarity(one_emb, word, dim=0)
                        # one_emb = torch.cat((senemb, word.unsqueeze(0)), dim=0)
                        # dianji = self.Rconv(one_emb.unsqueeze(0))
                        # dianji = (dianji.sum(-1))[0].item()
                        numcountSR[i][j] = dianji
            # _, SRid = torch.max(numcountSR, dim=1)
            # 最大的三个
            SRid = torch.sort(numcountSR, descending=True, dim=1)[1]
            SRid = SRid[:, 0:2].transpose(0, 1)  # [3 25]
            # WS = F.relu(torch.softmax(numcountSR, dim=1).unsqueeze(-1)-yuzhi)
            SRres = (Sout[tensor_rangeSR, SRid[0]] + Sout[tensor_rangeSR, SRid[1]])/2
            # SRres = torch.cat((Sout[tensor_rangeSR, SRid[0]], Sout[tensor_rangeSR, SRid[1]]), dim=1)


            for i in range(Qout.shape[0]):
                sentence = Qout[i]  # [128 768]
                senemb = Qemb[i]  # [3 768]
                # one_emb = QCLS[i]+senemb[2]
                # W_sub = (senemb[2] * (senemb[0] + senemb[1])).sum(0)
                # W_cls = (QCLS[i] * (senemb[0] + senemb[1])).sum(0)
                # W_S_C = torch.softmax(torch.Tensor([W_sub, W_cls]), dim=0)
                # one_emb = W_S_C[1] * QCLS[i] + W_S_C[0] * senemb[2]
                for j, word in enumerate(sentence[1:Qrela[i]]):
                        # one_emb = (senemb[0] + senemb[1])   ##T2
                        one_emb = senemb[2]
                        dianji = (one_emb * word).sum(0)
                        # dianji = torch.cosine_similarity(one_emb, word, dim=0)
                        # one_emb = torch.cat((senemb, word.unsqueeze(0)), dim=0)
                        # dianji = self.Rconv(one_emb.unsqueeze(0))
                        # dianji = (dianji.sum(-1))[0].item()
                        numcountQR[i][j] = dianji
            # _, QRid = torch.max(numcountQR, dim=1)
            # 最大的三个
            QRid = torch.sort(numcountQR, descending=True, dim=1)[1]
            QRid = QRid[:, 0:2].transpose(0, 1)  # [3 25]
            # WQ = F.relu(torch.softmax(numcountQR, dim=1).unsqueeze(-1)-yuzhi)
            QRres = (Qout[tensor_rangeQR, QRid[0]] + Qout[tensor_rangeQR, QRid[1]])/2
            # QRres = torch.cat((Qout[tensor_rangeQR, QRid[0]] , Qout[tensor_rangeQR, QRid[1]]),dim=1)


            support = torch.cat((support, SRres), dim=1)
            query = torch.cat((query, QRres), dim=1)

            hidden_size = support.size(-1)


            # K = 1
            support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
            query = query.view(-1, total_Q, hidden_size)  # (B, total_Q, D)

        if not target is None:
            # DB
            if N == 5:
                labeldb = torch.Tensor([0, 1, 2, 3, 4]).long().cuda()
            else:
                labeldb = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).long().cuda()
            # supportDB = support + support * torch.mean(self.dimdrop1d(target), dim=0)
            # queryDB = query + query * torch.mean(self.dimdrop1d(target), dim=0)

            # supportDB = torch.mean(self.dimdrop1d(support), 2)
            support = torch.mean(support, 2)
            # supportDB = torch.mean(supportDB, 2)
            # supportDB = support + support * self.dimdrop1d(target.unsqueeze(0))
            # supportDB = support + support * target.unsqueeze(0)
            supportDB = self.dimdrop1d(support)

            logitsdbS = self.__batch_dist__(supportDB, support, None)/0.5
            # logitsdbS = self.__batch_dist__(support, supportDB, None)
            # logitsdbQ = self.__batch_dist__(query, queryDB, None)
            # print(logitsdbQ)
            # print('')

            # dbloss = self.cost(logitsdbS.view(-1, N), labeldb.view(-1)) + self.cost(logitsdbQ.view(-1, N), labeldb.view(-1))
            dbloss = self.cost(logitsdbS.view(-1, N), labeldb.view(-1))
            # dbloss = self.cost(logitsdbQ.view(-1, N), labeldb.view(-1))

            # support = supportDB
            # query = queryDB

            corloss = corloss + dbloss * 2
        else:
            support = torch.mean(support, 2)  # Calculate prototype for each class

        # support = torch.mean(support, 2)

        if K == 1 and len(sne) < 100:
            sne.append(support.view(N, -1).data.cpu().numpy())
            if len(sne) == 100:
                print('saved')
                np.save('sne.npy', sne)

        logits = self.__batch_dist__(support, query, None)  # (B 1 N D)  (B totalQ 1 D) (B, total_Q, N)
        # print(torch.norm(support.squeeze(0), dim=1))
        # print(torch.norm(query.squeeze(0), dim=1))
        # print(logits)

        # if not target is None:
        #     logits = logits.view(-1, N)
        #     sftlogits = torch.softmax(logits,dim=1)
        #     medium_S = torch.einsum('ij,jk->ik', sftlogits, support.squeeze(0))
        #     # print(sftlogits)
        #     logits_fin = torch.ones_like(logits)
        #     for iid in range(N):
        #         logits_fin[iid] = self.__batch_dist__(support-medium_S[iid], (query[:, iid]-medium_S[iid]).unsqueeze(0), None).view(N)
        #         # logits_fin[iid] = self.__batch_dist__(support+torch.cat((featsuplist[iid], torch.zeros(768).cuda()),dim=-1).unsqueeze(0), (query[:, iid]+featsuplist[iid]).unsqueeze(0), None).view(N)
        #
        #     # print(logits_fin)
        #     # print('')
        #     logits = logits_fin

        logits = logits / 0.5
        _, pred = torch.max(logits.view(-1, N), 1)
        # print(logits.shape)
        # print(pred.shape)
        return logits, pred, (adv_S, adv_T, 0, 0, 0, corloss)

    # W1    1.0 1.0 1.0
    # W2    1.5 1.0 1.0
    # W3    2.0 1.0 1.0
    # W4    2.5 1.5 1.0

    # W8    1.5 0.5 1.0
    # W5    1.5 1.0 1.0
    # W6    1.5 1.5 1.0
    # W7    1.5 2.0 1.0



