#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:46:50 2020

@author: nirmalenduprakash
document classification using BERT pretrained model
"""
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.nn import functional as F    
from transformers import BertModel
import nltk
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import os

#faced trouble downloading pertrianed model on virtual device, so downloaded manually
bert_model=BertModel.from_pretrained('/home/svu/e0401988/NLP/classification')
tokenizer = BertTokenizer.from_pretrained('/home/svu/e0401988/NLP/classification')


'''preprocessed training data
encoded using BERT tokenizer
All promotional articles are labelled 1 and good articles 0
'''
with open('/home/svu/e0401988/NLP/classification/train.pkl','rb') as f:
  df=pickle.load(f)
cls_id= tokenizer.convert_tokens_to_ids('[CLS]')
df['cls_index']=df['encoding'].apply(lambda x: [1 if tok==cls_id else 0 for tok in x ])
  
class ClassifierDataset(Dataset):
  def __init__(self,df):
    self.df=df

  def __len__(self): 
    return len(self.df)

  def __getitem__(self,index):
    return torch.tensor(self.df.iloc[index]['encoding']),torch.tensor(self.df.iloc[index]['attn_mask']),\
      torch.tensor(self.df.iloc[index]['label']),torch.tensor(self.df.iloc[index]['cls_index'])

class DocumentClassifier(nn.Module):
    def __init__(self, freeze_bert = True):
        super(DocumentClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        for p in self.bert_layer.parameters():
            p.requires_grad = False
        
        #Classification layer
        self.cls_layer = nn.Linear(768, 1)
        # self.sigmoid=nn.Sigmoid()

    def forward(self, seq, attn_masks,cls_index):
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)
        # print(cont_reps.shape)
        # cls_rep = cont_reps[:, 0]
        batch_rep=[]
        for indx in range(seq.shape[0]):          
          cls_rep=torch.mean(cont_reps[indx,[i for i in cls_index[indx] if i==1]],dim=0).view(1,-1)        
          batch_rep.append(cls_rep)

        batch_rep=torch.cat(batch_rep,dim=0)          
        logits = self.cls_layer(batch_rep)
        return logits

#training and validation dataset        
msk = np.random.rand(len(df)) < 0.8
train=df[msk]
val=df[~msk]
train_set=ClassifierDataset(train)
val_set=ClassifierDataset(val)
train_loader=DataLoader(train_set, batch_size = 16)
val_loader = DataLoader(val_set, batch_size = 16)        

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():    
    model=DocumentClassifier().to(device)
    if(os.path.exists('/home/svu/e0401988/NLP/classification/model.pt')):
        model.load_state_dict(torch.load('/home/svu/e0401988/NLP/classification/model.pt'))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 2e-5)
    
    
    # training_loss=[]
    val_losses=[]
    if(os.path.exists('/home/svu/e0401988/NLP/classification/val_losses.pkl')):
      with open('/home/svu/e0401988/NLP/classification/val_losses.pkl','rb') as f:
        val_losses=pickle.load(f)
    
    for _e in range(20):
        train_loss=0
        for t, (seq, attn_mask, labels,cls_index) in enumerate(train_loader):
            # data_batch = sort_batch_by_len(data_dict)
            
            seq=seq.to(device)
            attn_mask=attn_mask.to(device)
            labels =labels.to(device) #torch.tensor(data_batch).to(device)
                    
            optimizer.zero_grad()
            logits=model(seq,attn_mask,cls_index)
            loss = criterion(logits.squeeze(-1), labels.float())
            train_loss+=loss.data.item()
            loss.backward()
            optimizer.step()
        train_loss= np.mean(train_loss)
        val_loss=0
        for t, (seq, attn_mask, labels,cls_index) in enumerate(val_loader):
            # data_batch = sort_batch_by_len(data_dict)
            
            seq=seq.to(device)
            attn_mask=attn_mask.to(device)
            labels =labels.to(device) #torch.tensor(data_batch).to(device)
                    
            optimizer.zero_grad()
            logits=model(seq,attn_mask,cls_index)
            loss = criterion(logits.squeeze(-1), labels.float())
            val_loss+=loss.data.item()
        val_loss= np.mean(val_loss)   
        if(len(val_losses)>0 and val_loss<min(val_losses)):
          torch.save(model.state_dict(), '/home/svu/e0401988/NLP/classification/model.pt')  
        val_losses.append(val_loss)      
        print('training loss:{} validation loss:{}'.format(train_loss,val_loss))
    
    with open('/home/svu/e0401988/NLP/classification/val_losses.pkl','wb') as f:
      pickle.dump(val_losses,f)        

def predict(doc):
    model=DocumentClassifier().to(device)
    if(os.path.exists('/home/svu/e0401988/NLP/classification/model.pt')):
        model.load_state_dict(torch.load('/home/svu/e0401988/NLP/classification/model.pt'))
    tokens=[['[CLS]']+tokenizer.tokenize(sent)+['[SEP]'] for sent in nltk.sent_tokenize(doc)]    
    if(len(tokens)<512):
        tokens+=['[PAD]' for _ in range(512-len(tokens))]
    else:
        tokens=tokens[:512]
    attn_mask=torch.tensor([0 if tok=='[PAD]' else 1 for tok in tokens]).to(device)
    cls_mask=torch.tensor([1 if tok==cls_id else 0 for tok in tokens]).to(device)    
    encoded=torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).to(device)
    output=model(encoded,attn_mask,cls_mask)
    return nn.sigmoid(output)>0.5

if  __name__== "__main__:
    train()
    
#preprocessing
df_promo=pd.read_csv('/content/drive/My Drive/wikipedia-promotional-articles/promotional.csv')
df_good=pd.read_csv('/content/drive/My Drive/wikipedia-promotional-articles/good.csv')
df_promo.drop(labels=['advert','coi','fanpov','pr','resume','url'],inplace=True,axis=1)
df_promo['label']=1
df_good.drop(labels=['url'],axis=1,inplace=True)
df_good['label']=0

df_good['encoding']=df_good['text'].apply(lambda x: [['[CLS]']+tokenizer.tokenize(sent)+['[SEP]'] for sent in nltk.sent_tokenize(x)])
df_promo['encoding']=df_promo['text'].apply(lambda x: [['[CLS]']+tokenizer.tokenize(sent)+['[SEP]'] for sent in nltk.sent_tokenize(x)])
df=pd.concat([df_good,df_promo],axis=0,join='inner')
def pad(tokens):
  if(len(tokens)<512):
    return tokens+['[PAD]' for _ in range(512-len(tokens))]
  else:
    return tokens[:512]  
df['encoding']=df['encoding'].apply(lambda x: pad([w for l in x for w in l]))
print(df.isna().sum())
df['attn_mask']=df['encoding'].apply(lambda x: [0 if tok==str('[PAD]') else 1 for tok in x ])
df['encoding']=df['encoding'].apply(lambda x:tokenizer.convert_tokens_to_ids(x))
with open('/content/drive/My Drive/wikipedia-promotional-articles/train.pkl','wb') as f:
  pickle.dump(df,f)        