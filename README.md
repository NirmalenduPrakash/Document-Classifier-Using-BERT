# Document Classification using BERT pretrained weights

```
Prerequistes
```
* torch
* transformers(pretrained model and config available here)
* nltk
* pickle

```Data
Kaggle(https://www.kaggle.com/urbanbricks/wikipedia-promotional-articles)
All promotional articles considered as one class, thus a binary classification problem
Eventually to be used as a classifier for any news article online.
Many articles online promote sponsored content without explicitly mentioning it. This model can help users.
```

```
working
```
![Screen Shot 2020-03-14 at 12 40 38 PM](https://user-images.githubusercontent.com/19767662/76675102-0db96900-65f1-11ea-8b01-c6aa65645ab3.png)

'[CLS]' must represent sentence embedding, as it always occurs at the start of the sentence, unlike 
other tokens which appear based on english grammar, and it's attention is calculated with all other tokens of the 
sentence.
Since, it is a document classification task, we can safely take an average across all CLS tokens, for a
document.
```
Steps
```
* Change the train-validation file paths based on your system directory.
* BERT model can be downloaded separately and then read from directory manually or just installed using package name
* To predict, pass the read document to the predict method, it uses a threshold of 0.5, can be changed.
Also, the document is trimmed to 512 characters, as this is the max set by BERT 


```
How it works
```

* Encoder model with BERT encoder.
* The first index encoded output from BERT layer, which represents sentence encoding is passed through a linear layer.
* BERT layer is fixed. Only the linear layer weights are adjusted during training.

```
Validation
To be updated(Training in progress)
```
```
Future Enhancements
```
* To be updated.


## Any questions 👨‍💻
<p> If you have any questions, feel free to ask me: </p>
<p> 📧: "nirmalendu@outlook.com"></p>
