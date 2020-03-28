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
bert_model=BertModel.from_pretrained('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload')
tokenizer = BertTokenizer.from_pretrained('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload')


'''preprocessed training data
encoded using BERT tokenizer
All promotional articles are labelled 1 and good articles 0
'''
with open('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/train.pkl','rb') as f:
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
 
        
        batch_rep=[]
        for indx in range(seq.shape[0]):          
          cls_rep=torch.mean(cont_reps[indx,[i for i in cls_index[indx] if i==1]],dim=0).view(1,-1)        
          batch_rep.append(cls_rep)                  
        
        batch_rep=torch.cat(batch_rep,dim=0)   
#        print(batch_rep.data.numpy().sum())
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

class classifier():
    def __init__(self):
       self.model= DocumentClassifier().to(device)
       if(os.path.exists('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/model.pt')):
            self.model.load_state_dict(torch.load('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/model.pt'))
       self.sigmoid=nn.Sigmoid()     
    def train(self):    
        model=DocumentClassifier().to(device)
        if(os.path.exists('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/model.pt')):
            model.load_state_dict(torch.load('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/model.pt'))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr = 2e-5)
        
        
        # training_loss=[]
        val_losses=[]
        if(os.path.exists('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/val_losses.pkl')):
          with open('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/val_losses.pkl','rb') as f:
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
              torch.save(model.state_dict(), '/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/model.pt')  
            val_losses.append(val_loss)      
            print('training loss:{} validation loss:{}'.format(train_loss,val_loss))
        
        with open('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/val_losses.pkl','wb') as f:
          pickle.dump(val_losses,f)        
    
    def predict(self,doc):
#        model=DocumentClassifier().to(device)
#        if(os.path.exists('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/model.pt')):
#            model.load_state_dict(torch.load('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/model.pt'))
        tokens=[['[CLS]']+tokenizer.tokenize(sent)+['[SEP]'] for sent in nltk.sent_tokenize(doc)] 
        tokens=[t for tok in tokens for t in tok]
        if(len(tokens)<512):
            tokens+=['[PAD]' for _ in range(512-len(tokens))]
        else:
            tokens=tokens[:512]
            
        attn_mask=torch.tensor([0 if tok=='[PAD]' else 1 for tok in tokens]).view(1,-1).to(device)
        cls_mask=torch.tensor([1 if tok=='[CLS]' else 0 for tok in tokens]).view(1,-1).to(device)    
        encoded=torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).view(1,-1).to(device)

        output=self.model(encoded,attn_mask,cls_mask)

        return (self.sigmoid(output)>0.5).data.item(),self.sigmoid(output).data.item()

if  __name__== "__main__":
    train()
    
#preprocessing
def preprocess():    
    df_promo=pd.read_csv('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/promotional.csv')
    df_good=pd.read_csv('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/good.csv')
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

    df['attn_mask']=df['encoding'].apply(lambda x: [0 if tok==str('[PAD]') else 1 for tok in x ])
    df['encoding']=df['encoding'].apply(lambda x:tokenizer.convert_tokens_to_ids(x))
    with open('/Users/nirmalenduprakash/Documents/Project/NLP/classification/upload/train.pkl','wb') as f:
      pickle.dump(df,f)        