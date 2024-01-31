import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import json
import math
import random

import numpy as np
import torch


# path = '../data/cn_relations_mix.txt'
# # path = '../data/cn_relations_orig.txt'
# # path = '../data/latest-lexemes.json'
# cndata = {}
# cnt = 0
# for triple in open(path, 'r', encoding='utf-8'):
#     arg11, r, arg22 = triple.split(', ')
#     arg1 = arg11.strip().lower()
#     r = r.strip()
#     arg2 = arg22.strip().lower()
#     arg1 = arg1.replace('_', ' ')
#     arg2 = arg2.replace('_', ' ')
#     if arg1==arg2:
#         continue
#     # print(arg1, r, arg2)
#     if not arg1 in cndata:
#         cndata[arg1] = {}
#     cndata[arg1][arg2] = r
#     if not arg2 in cndata:
#         cndata[arg2] = {}
#     cndata[arg2][arg1] = r
#     cnt += 1
# print(cnt)
# np.save('cndata.npy', cndata)
# print('saved')




cndata = np.load('cndata.npy', allow_pickle=True).item()
print('load cndata')

def getrela1(w1, w2):
    rela = []
    if w1 in cndata:
        re = cndata[w1].get(w2, '')
        if re != '' and not re in rela:
            rela.append(re)
    return rela

def getrela2(w1, w2):
    rela = getrela1(w1, w2)
    if not rela:
        w1 = w1.split(' ')
        w2 = w2.split(' ')
        for m in w1:
            if m in cndata:
                for n in w2:
                    re = cndata[m].get(n, '')
                    if re != '' and not re in rela:
                        rela.append(re)
                        # return rela

    return rela

def getT(w):
    nbw = list(cndata[w])
    tw = []
    # print(w,nbw)

    # if len(nbw)<10:
    #     selectnb = -1
    #     maxD = -1
    #     minD = 9999999999
    #
    #     for nb in nbw:  #Max
    #         if len(cndata[nb])>maxD:
    #             maxD = len(cndata[nb])
    #             selectnb = nb
    #
    #     # for nb in nbw:  #Min
    #     #     if len(cndata[nb])<minD:
    #     #         minD = len(cndata[nb])
    #     #         selectnb = nb
    #
    #     tw.append(selectnb)
    #     return tw
    # if len(nbw)<2:
    #     tw.append(nbw[0])
    #     return tw

    # if len(nbw)>100:
    #     selectnb = -1
    #     maxD = -1
    #     minD = 9999999999
    #
    #     # for nb in nbw:  #Max
    #     #     if len(cndata[nb])>maxD:
    #     #         maxD = len(cndata[nb])
    #     #         selectnb = nb
    #     # tw.append(selectnb)
    #     #
    #     # for nb in nbw:  #Min
    #     #     if len(cndata[nb])<minD:
    #     #         minD = len(cndata[nb])
    #     #         selectnb = nb
    #     # if not selectnb in tw:
    #     #     tw.append(selectnb)
    #
    # elif len(nbw)>0:
    #     selectnb = -1
    #     maxD = -1
    #     minD = 9999999999
    #
    #     # for nb in nbw:  # Max
    #     #     if len(cndata[nb]) > maxD:
    #     #         maxD = len(cndata[nb])
    #     #         selectnb = nb
    #     # tw.append(selectnb)
    #
    #     num = [0] * len(nbw)
    #     for i in range(len(nbw)):
    #         for j in range(i + 1, len(nbw)):
    #             if cndata[nbw[i]].get(nbw[j], '') != '':
    #                 num[i] += 1
    #                 num[j] += 1
    #     id1 = num.index(max(num))
    #     if not nbw[id1] in tw:
    #         tw.append(nbw[id1])
    #     # id2 = num.index(min(num))
    #     # tw.append(nbw[id2])
    # # else:
    # #     num = [0] * len(nbw)
    # #     for i in range(len(nbw)):
    # #         for j in range(i + 1, len(nbw)):
    # #             if cndata[nbw[i]].get(nbw[j], '') != '':
    # #                 num[i] += 1
    # #                 num[j] += 1
    # #     id1 = num.index(min(num))
    # #     if not nbw[id1] in tw:
    # #         tw.append(nbw[id1])


    if len(nbw)>100:
        nbw = nbw[0:100]
    selectnb = -1
    maxD = -1
    minD = 9999999999

    # for nb in nbw:  # Max
    #     if len(cndata[nb]) > maxD:
    #         maxD = len(cndata[nb])
    #         selectnb = nb
    # tw.append(selectnb)

    num = [0] * len(nbw)
    for i in range(len(nbw)):
        for j in range(i + 1, len(nbw)):
            if cndata[nbw[i]].get(nbw[j], '') != '':
                num[i] += 1
                num[j] += 1
    id1 = num.index(max(num))
    tw.append(nbw[id1])
    id2 = num.index(min(num))
    tw.append(nbw[id2])

    return tw

# def getT(w):
#     nbw = list(cndata[w])
#     tw = set()
#     # print(w,nbw)
#     if len(nbw)==1:
#         tw.add(nbw[0])
#         return tw
#     # if len(nbw)<5:
#     #     tw.add(nbw[0])
#         # return tw
#
#     if len(nbw)>100:
#         nbw = nbw[0:100]
#     num = [0] * len(nbw)
#     for i in range(len(nbw)):
#         for j in range(i + 1, len(nbw)):
#             if cndata[nbw[i]].get(nbw[j], '') != '':
#                 num[i] += 1
#                 num[j] += 1
#     id1 = num.index(max(num))
#     tw.add(nbw[id1])
#     num[id1] = -1
#     id2 = num.index(max(num))
#
#     # tw.add(nbw[id2])
#     # print(tw)
#     # if w==tw:
#     #     print(w)
#     #     print(tw)
#     #     print('')
#     return tw

def getallrela(w1, w2): #获得原实体关系 or 分割实体关系 or  替换原实体关系 or  替换分割实体关系
    rela = getrela2(w1, w2)   #原实体关系 分割实体关系
    tw1 = []
    tw2 = []
    if not rela:       #  替换原实体关系
        if w1 in cndata:
            tw1 = getT(w1)
        if w2 in cndata:
            tw2 = getT(w2)

        # #RallT
        # for t1 in tw1:
        #     for t2 in tw2:
        #         re = cndata[t1].get(t2, '')  # 原实体都在网络中  且其对应的替换原实体有关系  则返回关系
        #         if re != '' and not re in rela:
        #             rela.append(re)

        # Rall
        # if  tw1  and not tw2:
        #     re = cndata[tw1[0]].get(w2, '')  # 原实体都在网络中  且其对应的替换原实体有关系  则返回关系
        #     if re != '' and not re in rela:
        #         rela.append(re)
        # if  not tw1  and tw2:
        #     re = cndata[tw2[0]].get(w1, '')  # 原实体都在网络中  且其对应的替换原实体有关系  则返回关系
        #     if re != '' and not re in rela:
        #         rela.append(re)
        # if  tw1  and tw2:
        #     re = cndata[tw1[0]].get(tw2[0], '')  # 原实体都在网络中  且其对应的替换原实体有关系  则返回关系
        #     if re != '' and not re in rela:
        #         rela.append(re)

        #oneR
        if  tw1  and not tw2:  #R0
            re = cndata[tw1[0]].get(w2, '')  # 原实体都在网络中  且其对应的替换原实体有关系  则返回关系
            if re != '' and not re in rela:
                rela.append(re)
        if  not rela and not tw1  and tw2:
            re = cndata[tw2[0]].get(w1, '')  # 原实体都在网络中  且其对应的替换原实体有关系  则返回关系
            if re != '' and not re in rela:
                rela.append(re)
        if not rela and  tw1  and tw2:   #R1
            re = cndata[tw1[0]].get(tw2[0], '')  # 原实体都在网络中  且其对应的替换原实体有关系  则返回关系
            if re != '' and not re in rela:
                rela.append(re)
    #
    if not rela or True: #替换分割实体关系   TFR
        w1 = w1.split(' ')
        w2 = w2.split(' ')
        if not tw1:
            for e1 in w1:
                if e1 in cndata:
                    tw1 += getT(e1)
        if not tw2:
            for e2 in w2:
                if e2 in cndata:
                    tw2 += getT(e2)
        for e1 in tw1:
            for e2 in tw2:
                re = cndata[e1].get(e2, '')
                if re != '' and not re in rela:
                    rela.append(re)    #0
                    # return rela      #1
    return rela   #set()

def getallTst(w1, w2):  # 替换原实体 or  替换分割实体
    tw1 = []
    tw2 = []
    if w1 in cndata:
        tw1 = tw1+getT(w1)
    if w2 in cndata:
        tw2 = tw2+getT(w2)
    # if not tw1:
    #     w1 = w1.split(' ')
    #     for e1 in w1:
    #         if e1 in cndata:
    #             tw1 = getT(e1)
    #             # break
    #
    # if not tw2:
    #     w2 = w2.split(' ')
    #     for e2 in w2:
    #         if e2 in cndata:
    #             tw2 = getT(e2)
    #             # break
    return (tw1, tw2)



