#!/usr/bin/env python
# coding: utf-8

# In[103]:


#Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from tqdm import tqdm
import random 
import string
import re
import gensim
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import punkt
stop_words = stopwords.words('english')
from nltk.util import ngrams
from collections import defaultdict
from collections import Counter
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[33]:


#Read the data
train= pd.read_csv('C:/Users/priya/Downloads/Python/nlp-getting-started/train.csv').fillna(' ')
test=pd.read_csv('C:/Users/priya/Downloads/Python/nlp-getting-started/test.csv').fillna(' ')
train.head(3)


# In[7]:


#EXPLORATORY DATA ANALYSIS

#Frequency of disaster and non-disaster
x=train.target.value_counts()

barlist = plt.bar(x.index,x)
barlist[0].set_color('b')
barlist[1].set_color('r')
bars=('0','1')
plt.xticks(x.index, bars)
plt.title("Frequency of disaster vs non disaster tweets")
plt.show()
x


# In[8]:


#Characters in tweets

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=train[train['target']==1]['text'].str.len()
ax1.hist(tweet_len,color='red')
ax1.set_title('disaster tweets')

tweet_len=train[train['target']==0]['text'].str.len()
ax2.hist(tweet_len,color='blue')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.show()


# In[9]:


#Words in the tweet

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=train[train['target']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='red')
ax1.set_title('disaster tweets')

tweet_len=train[train['target']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='blue')
ax2.set_title('Not disaster tweets')
fig.suptitle('Words in a tweet')
plt.show()


# In[30]:


#Hashtag analysis
def show_word_distrib(target=1, field="text"):
    txt = df[df['target']==target][field].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(txt)
    words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stop_words) 
    
    rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N),
                        columns=['Word', 'Frequency']).set_index('Word')
    print(rslt)

print("-Hashtag Analysis ")
def find_hashtags(train):
    return ", ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", train)]) or None
def add_hashtags(df):
    from sklearn.feature_extraction.text import CountVectorizer
    
    df['hashtag'] = df["text"].apply(lambda x: find_hashtags(x))
    df['hashtag'].fillna(value="no", inplace=True)
    
    return df
    
top_N = 20
df = add_hashtags(train)

length = len([v for v in df.hashtag.values if isinstance(v, str)])
print("-Number of tweets with hashtags: {}".format(length))
print("-- Hashtag distribution in disaster samples ")
show_word_distrib(target=1, field="hashtag")
print("-- Hashtag distribution in non-disaster samples ")
show_word_distrib(target=0, field="hashtag")


# In[110]:


df_test = add_hashtags(test)
df_test


# In[15]:


# Creating corpus for analysis of word patterns
def create_corpus(target):
    corpus=[]
    
    for x in train[train['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# In[23]:


#Frequency of stopwords
corpus=create_corpus(0)

dic=defaultdict(int)
for word in corpus:
    if word in stop_words:
        dic[word]+=1
        
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]

x,y=zip(*top)
plt.title('Frequency of stopwords for non-disaster tweets')
plt.bar(x,y, color= 'blue')


# In[22]:


corpus=create_corpus(1)

dic=defaultdict(int)
for word in corpus:
    if word in stop_words:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]

x,y=zip(*top)
plt.title('Frequency of stopwords for disaster tweets')
plt.bar(x,y,color='red')


# In[20]:


#Frequency of special characters
plt.figure(figsize=(10,5))
corpus=create_corpus(1)

dic=defaultdict(int)
import string
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
        
x,y=zip(*top)
plt.title('Frequency of special characters in Disaster tweets')
plt.bar(x,y,color='r')


# In[24]:


plt.figure(figsize=(10,5))
corpus=create_corpus(0)

dic=defaultdict(int)
import string
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
x,y=zip(*top)
plt.title("Frequency of special characters for non disaster tweets")
plt.bar(x,y,color='blue')


# In[31]:


#Analysis of keywords

disaster_keywords = [kw for kw in train.loc[df.target == 1].keyword]
regular_keywords = [kw for kw in train.loc[df.target == 0].keyword]

disaster_keywords_counts = dict(pd.DataFrame(data={'x': disaster_keywords}).x.value_counts())
regular_keywords_counts = dict(pd.DataFrame(data={'x': regular_keywords}).x.value_counts())
all_keywords_counts =  dict(pd.DataFrame(data={'x': df.keyword.values}).x.value_counts())

# we sort the keywords so the most frequents are on top and we print them with relative
# occurrences in both classes of tweets:
for keyword, _ in sorted(all_keywords_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print("> KW: {}".format(keyword))
    print("-- # in disaster tweets: {}".format(disaster_keywords_counts.get(keyword, 0)))
    print("-- # in non-disaster tweets: {}".format(regular_keywords_counts.get(keyword, 0)))
    print('--------')


# In[58]:


#Cleaning the data   
    
#Removing URLs
    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)
    df['text']=df['text'].apply(lambda x : remove_URL(x))
    
#Removing HTML
    def remove_html(text):
        html=re.compile(r'<.*?>')
        return html.sub(r'',text)
    df['text']=df['text'].apply(lambda x : remove_html(x))

#Removing emoticons
    def remove_emoji(text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    df['text']=df['text'].apply(lambda x: remove_emoji(x))

#Removing Punctuations
    def remove_punct(text):
        table=str.maketrans('','',string.punctuation)
        return text.translate(table)
    df['text']=df['text'].apply(lambda x : remove_punct(x))
    
    df.text = df.text.replace('\s+', ' ', regex=True)
    return df

#Cleaning
df = clean_df(df)
print("-- Word distrig Disaster Class")
show_word_distrib(target=1, field="text")
print("-- Word distrib Non Disaster Class")
show_word_distrib(target=0, field="text")


# In[112]:


#Also cleaning data for actual test- submission data
df_test = clean_df(df_test)


# In[41]:


N=10

#Analyzing n-grams
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in stop_words]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

from collections import deque

# Bigrams
disaster_bigrams = defaultdict(int)
nondisaster_bigrams = defaultdict(int)

for txt in df[df['target']==1]['text']:
    for word in generate_ngrams(txt, n_gram=2):
        disaster_bigrams[word] += 1
        
for txt in df[df['target']==0]['text']:
    for word in generate_ngrams(txt, n_gram=2):
        nondisaster_bigrams[word] += 1
        
df_disaster_bigrams = pd.DataFrame(sorted(disaster_bigrams.items(), key=lambda x: x[1])[::-1])
df_nondisaster_bigrams = pd.DataFrame(sorted(nondisaster_bigrams.items(), key=lambda x: x[1])[::-1])

# Trigrams
disaster_trigrams = defaultdict(int)
nondisaster_trigrams = defaultdict(int)

for txt in df[df['target']==1]['text']:
    for word in generate_ngrams(txt, n_gram=3):
        disaster_trigrams[word] += 1
        
for txt in df[df['target']==0]['text']:
    for word in generate_ngrams(txt, n_gram=3):
        nondisaster_trigrams[word] += 1
        
df_disaster_trigrams = pd.DataFrame(sorted(disaster_trigrams.items(), key=lambda x: x[1])[::-1])
df_nondisaster_trigrams = pd.DataFrame(sorted(nondisaster_trigrams.items(), key=lambda x: x[1])[::-1])


# In[42]:


#Vizualizing Bigrams
fig, axes = plt.subplots(ncols=2, figsize=(18, 10), dpi=100)
plt.tight_layout()

sns.barplot(y=df_disaster_bigrams[0].values[:N], x=df_disaster_bigrams[1].values[:N], ax=axes[0], color='red')
sns.barplot(y=df_nondisaster_bigrams[0].values[:N], x=df_nondisaster_bigrams[1].values[:N], ax=axes[1], color='blue')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=13)

axes[0].set_title(f'Top {N} most common bigrams in Disaster Tweets', fontsize=15)
axes[1].set_title(f'Top {N} most common bigrams in Non-disaster Tweets', fontsize=15)

plt.show()


# In[43]:


#Vizualizing Trigrams
fig, axes = plt.subplots(ncols=2, figsize=(20, 10), dpi=100)

sns.barplot(y=df_disaster_trigrams[0].values[:N], x=df_disaster_trigrams[1].values[:N], ax=axes[0], color='red')
sns.barplot(y=df_nondisaster_trigrams[0].values[:N], x=df_nondisaster_trigrams[1].values[:N], ax=axes[1], color='blue')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=11)

axes[0].set_title(f'Top {N} most common trigrams in Disaster Tweets', fontsize=15)
axes[1].set_title(f'Top {N} most common trigrams in Non-disaster Tweets', fontsize=15)

plt.show()


# In[115]:


#MODEL BUILDING

#Predictor and response
X = df['text']
y = df['target']

#Submission data
X_sub = df_test['text']

#Set seed before split
random.seed(13258676)

#Splitting cleaned data to train and test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

all_text = pd.concat([X_train, X_test])

print("Checkpoint1 - Data Read Complete")


# In[60]:


#Word embedding using GloVe
embeddings_index = {}
f = open('C:/Users/priya/Downloads/Python/glove.6B/glove.6B.300d.txt','r', encoding="utf-8")
for line in tqdm(f):
    values = line.split()
    word = values[0]
    try:
       coefs = np.asarray(values[1:], dtype='float32')
       embeddings_index[word] = coefs
    except ValueError:
       pass
f.close()
print('Found %s word vectors.' % len(embeddings_index))
# this function creates a normalized vector for the whole sentence
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

# create sentence vectors using the above function for training and validation set
xtrain_glove = [sent2vec(x) for x in tqdm(X_train)]
xtest_glove = [sent2vec(x) for x in tqdm(X_test)]
xfull_glove = [sent2vec(x) for x in tqdm(X)]


# In[117]:


#Vectorizing submission data
xsub_glove = [sent2vec(x) for x in tqdm(X_sub)]
xsub_glove_new = np.array(xsub_glove)


# In[61]:


xtrain_glove_new = np.array( xtrain_glove)
xtest_glove_new = np.array(xtest_glove)
xfull_glove_new = np.array(xfull_glove)


# In[95]:


# Fitting Logistic Regression Model to the Training set
classifier_lr = LogisticRegression(class_weight=({0:0.8,1:0.2}))
classifier_lr.fit(xtrain_glove_new, y_train)
# Predicting the Train data set results
y_train_lr = classifier_lr.predict(xtrain_glove_new)
y_pred_lr = classifier_lr.predict(xtest_glove_new)
# Making the Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_lr

y_pred_prob = classifier_lr.predict_proba(xtest_glove_new)[:, 1]
cm = pd.crosstab(y_pred_prob>0.25, y_test)
cm_lr


# In[94]:


#Calculating Model Accuracy
print('Logistic Regression Model Accuracy for Train Data set is {}'.format(accuracy_score(y_train_lr, y_train)))
print('Logistic Regression Model Accuracy for Test Data set is {}'.format(accuracy_score(y_pred_lr, y_test)))
print('Logistic Regression Model F1 Score is {}'.format(f1_score(y_test, y_pred_lr)))

cv_score = cross_val_score(classifier_lr,xfull_glove, y, cv=5, scoring='f1_macro')
cv_score
np.mean(cv_score)


# In[91]:


#KNN- Parameter tuning
neighbors = list(range(1,15))
train_knn_results = []
test_knn_results = []
for n in neighbors:
   model = KNeighborsClassifier(n_neighbors=n)
   model.fit(xtrain_glove_new, y_train)
   train_lr = model.predict(xtrain_glove_new)
   pred_lr = model.predict(xtest_glove_new)
   train_knn_results.append(accuracy_score(train_lr, y_train))
   test_knn_results.append(accuracy_score(pred_lr, y_test))


# In[92]:


from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(neighbors, train_knn_results, 'b', label="Train Accuracy")
line2, = plt.plot(neighbors, test_knn_results, 'r', label="Test Accuracy")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.show()


# In[74]:


# Fitting K- Nearest neighbour to the Training set
classifier_knn = KNeighborsClassifier(n_neighbors = 9 )
classifier_knn.fit(xtrain_glove_new, y_train)
# Predicting the Train data set results
y_train_knn = classifier_knn.predict(xtrain_glove_new)
y_pred_knn = classifier_knn.predict(xtest_glove_new)
# Making the Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_knn


# In[75]:


#Calculating Model Accuracy
print('K-Nearest Neighbour Model Accuracy for Train Data set is {}'.format(accuracy_score(y_train_knn, y_train)))
print('K-Nearest Neighbour Model Accuracy for Test Data set is {}'.format(accuracy_score(y_pred_knn, y_test)))
print('K-Nearest Neighbour Model F1 Score is {}'.format(f1_score(y_test, y_pred_knn)))

cv_score = cross_val_score(classifier_knn,xfull_glove_new, y, cv=5, scoring='f1_macro')
cv_score
np.mean(cv_score)


# In[ ]:


#Parameter tunning for Decision trees
max_depth = []
acc_gini = []
acc_entropy = []
for i in range(1,30):
 dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)
 dtree.fit(xtrain_glove_new, y_train)
 pred = dtree.predict(xtest_glove_new)
 acc_gini.append(accuracy_score(y_test, pred))
 ####
 dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
 dtree.fit(xtrain_glove_new, y_train)
 pred = dtree.predict(xtest_glove_new)
 acc_entropy.append(accuracy_score(y_test, pred))
 ####
 max_depth.append(i)

d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 
 'acc_entropy':pd.Series(acc_entropy),
 'max_depth':pd.Series(max_depth)})


# In[ ]:


# visualizing changes in parameters
plt.plot('max_depth','acc_gini', data=d, label='gini')
plt.plot('max_depth','acc_entropy', data=d, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()


# In[96]:


# Fitting Decision Tree Models to the Training set
classifier_dt = DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 5)           
classifier_dt.fit(xtrain_glove_new, y_train)
# Predicting the Train data set results
y_train_dt = classifier_dt.predict(xtrain_glove_new)
y_pred_dt = classifier_dt.predict(xtest_glove_new)
# Making the Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_dt


# In[98]:


#Calculating Model Accuracy
print('DecisionTree Model Accuracy for Train Data set is {}'.format(accuracy_score(y_train_dt, y_train)))
print('DecisionTree Model Accuracy for Test Data set is {}'.format(accuracy_score(y_pred_dt, y_test)))
print('DecisionTree Model F1 Score is {}'.format(f1_score(y_test, y_pred_dt)))

cv_score = cross_val_score(classifier_dt, xtrain_glove_new, y_train, cv=5, scoring='f1_macro')
cv_score
np.mean(cv_score)


# In[79]:


# Fitting Gradient Boosting Models to the Training set
classifier_gb = GradientBoostingClassifier(loss = 'deviance',
                                                   learning_rate = 0.01,
                                                   n_estimators = 100,
                                                   max_depth = 30,
                                                   random_state=55)
classifier_gb.fit(xtrain_glove_new, y_train)
# Predicting the Test data set results
y_pred_gb = classifier_gb.predict(xtest_glove_new)
# Making the Confusion Matrix
cm_gb = confusion_matrix(y_test, y_pred_gb)
cm_gb


# In[99]:


#Calculating Model Accuracy
# Predicting the Train data set results
y_train_gb = classifier_gb.predict(xtrain_glove_new)
print('Gradient Boosting Classifier Accuracy is {} for Train Data Set'.format(accuracy_score(y_train_gb, y_train)))
print('Gradient Boosting Classifier Accuracy is {} for Test Data Set'.format(accuracy_score(y_pred_gb, y_test)))
print('Gradient Boosting Classifier F1 Score is {} '.format(f1_score(y_test, y_pred_gb)))


# In[ ]:


#Parameter tuning for XGBOOST
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')

# grid search
model = XGBClassifier()
n_estimators = [100, 200, 300, 400, 500]
learning_rate = [0.0001, 0.001, 0.01, 0.1]

param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(xtrain_glove_new, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


# plot results
scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
for i, value in enumerate(learning_rate):
    plt.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Log Loss')
plt.savefig('n_estimators_vs_learning_rate.png')


# In[100]:


# Fitting XGBoost Model to the Training set
classifier_xgb = XGBClassifier(max_depth=6,learning_rate=0.01,n_estimators=500,objective='binary:logistic')
classifier_xgb.fit(xtrain_glove_new, y_train)
# Predicting the Train data set results
y_pred_xgb = classifier_xgb.predict(xtest_glove_new)
# Making the Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_xgb


# In[101]:


print('XG Boost Model Accuracy Score for Train Data set is {}'.format(classifier_xgb.score(xtrain_glove_new, y_train)))
print('XG Boost Model Accuracy Score for Test Data set is {}'.format(classifier_xgb.score(xtest_glove_new, y_test)))
print('XG Boost Model F1 Score is {}'.format(f1_score(y_test, y_pred_xgb)))


# In[105]:


#Voting classifier
# Fitting Logistic Regression Model to the Training set
models = [('K-Nerarest Neighbour', classifier_knn),
          ('LogisticRegression',classifier_lr),
          ('DecisionTree Classifier',classifier_dt),
          ('XGBoost Classifier',classifier_xgb),
          ('Gradient Boosting',classifier_gb)]
classifier_vc = VotingClassifier(voting = 'hard',estimators= models)
classifier_vc.fit(xtrain_glove_new, y_train)
# Predicting the Train data set results
y_pred_vc = classifier_dt.predict(xtest_glove_new)
# Making the Confusion Matrix
cm_vc = confusion_matrix(y_test, y_pred_vc)
cm_vc


# In[108]:


#Calculating Model Accuracy
print('Voting Classifier Model Accuracy Score for Train Data set is {}'.format(classifier_vc.score(xtrain_glove_new, y_train)))
print('Voting Classifier Model Accuracy Score for Test Data set is {}'.format(classifier_vc.score(xtest_glove_new, y_test)))
print('Voting Classifier Model F1 Score is {}'.format(f1_score(y_test, y_pred_vc)))


# In[124]:


#Predicting output for submission data
y_sub_pred_xgb = classifier_xgb.predict(xsub_glove_new)

#Fetching Id to differnt frame
y_sub_id= test[['id']]
#Converting Id into array
y_sub_id= y_sub_id.values
#Converting 2 dimensional y_test_id into single dimension 
y_sub_id=y_sub_id.ravel()
#Converting 2 dimensional y_test_pred for all predicted results into single dimension 
y_sub_pred_xgb= y_sub_pred_xgb.ravel()

submission_df_xgb=pd.DataFrame({"id":y_sub_id,"target":y_sub_pred_xgb})
submission_df_xgb

submission_df_xgb.set_index("id")
submission_df_xgb.to_csv("submission_xgb.csv",index=False)

