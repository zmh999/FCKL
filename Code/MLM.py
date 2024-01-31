import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# import logging
#
# logging.basicConfig(level=logging.INFO)
#
# Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# model = BertForMaskedLM.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('../pretrain/bert-base-uncased/vocab.txt')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
def getword(text):

    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)

    masked_index = tokenized_text.index('[MASK]')
    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    # predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
    # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    predicted_index = predictions[0][0][masked_index]
    predicted_token = []
    for k in range(5):
        id = torch.argmax(predicted_index).item()
        predicted_index[id] = -999999999
        predicted_token += tokenizer.convert_ids_to_tokens([id])
    return predicted_token



text1 = '[CLS] These tumors are the most common non-epithelial neoplasms of gastric wall. "common non-epithelial neoplasms" means [MASK]. [SEP]'
# text2 = '[CLS] These [unused0] are the most common non-epithelial neoplasms of gastric wall. "tumors" means [MASK] with "non-epithelial neoplasms" in this sentence [SEP]'
# cancer
#Newton served as the president of the Royal Society.
text2 = '[CLS] Newton served as the president of the Royal Society. "Newton" means [MASK]. [SEP]'   #scientist
text3 = '[CLS] Newton served as the president of the Royal Society. "the Royal Society" means [MASK]. [SEP]'  #academy
# text3 = '[CLS] Aniridia is a rare [unused0] ocular disorder of complete or partial iris hypoplasia. "congenital" means [MASK] with "iris hypoplasia" in this sentence [SEP]'
# text4 = '[CLS] Aniridia is a rare congenital ocular disorder of complete or partial [unused1]. "iris hypoplasia" means [MASK] with "congenital" in this sentence [SEP]'
# #
# text5 = '[CLS] The lateral lesions and [unused0], especially radicular cysts, are compared. "dental cysts" means [MASK] with "cradicular cysts" in this sentence [SEP]'
# text6 = '[CLS] The lateral lesions and dental cysts, especially [unused1], are compared. "cradicular cysts" means [MASK] with "dental cysts" in this sentence [SEP]'
#
# text7 = 'Jingkou District is one of three districts of [unused0] , Jiangsu province, China. "Zhenjiang" means [MASK] with "Jiangsu" in this sentence [SEP]'
# text8 = 'Jingkou District is one of three districts of Zhenjiang , [unused1] province, China. "Jiangsu" means [MASK] with "Zhenjiang" in this sentence [SEP]'
#
# alltext = [["\"tumors\"", "\"non-epithelial neoplasms\""], ["\"congenital\"", "\"iris hypoplasia\""] ,["\"dental cysts\"", "\"cradicular cysts\""], ['\"Zhenjiang\"','\"Jiangsu\"']]
# alltext = [["tumors", "non-epithelial neoplasms"], ["congenital", "iris hypoplasia"] ,["dental cysts", "cradicular cysts"], ['Zhenjiang','Jiangsu']]

# t1 = ["at", "the", "age", "59", ",", "her", "diabetes", "mellitus", "manifested", "with", "type", "2", "diabetic", "phenotype", ",", "but", "based", "on", "gad", "positivity", "later", "was", "reclassified", "as", "type", "1", "diabetes", "."]
# t1 = ' '.join(t1) + ' \'diabetes means [MASK].'
# t1 = ' '.join(t1) + '\' decompensated cirrhosis\' means [MASK]'
# print(t1)
# for t in alltext:
print(getword(text1))
print(getword(text2))
print(getword(text3))

['tumor', 'cancer', 'tumors', 'benign', 'no']
['newton', 'scientist', 'mathematician', 'astronomer', 'philosopher']
['society', 'science', 'academy', 'scientific', 'university']



# sen = "These tumors are the most common non-epithelial neoplasms of gastric wall."
# e1 = "\"tumors\""
# e2 = "\"non-epithelial neoplasms\""
# print(getword(sen, e1))
# print(getword(sen, e1))
# print(getword(sen, e2))
# print(getword(sen, e1))
# print(getword(sen, e2))