# encoding=utf8

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import os
from gensim import corpora, models, similarities, summarization
from six import iteritems
import uniout
from time import gmtime, strftime
import operator

########################

input_train_data_file = 'dataset/lyrics.dataset'

dictionary = corpora.Dictionary(document.split() for document in open(input_train_data_file))
stoplist = set("再 又 到 很 我 你 妳 我們 在 了 都 不 的 是 有 沒有 要 不要 讓 也 人 說 就 著 這 他 她 祢 阮 攏 咱 那 啦 喔 啊 吧 嗎 嘩啦啦 會 能 卻 這樣 和 好 做 詞 曲 ㄟ = % - ， ’ ( ) （ ） : ： . 。 ！ ! 、 ～ ~ / i you the to me my be don't it your and t a s oh go da la".split())
#一樣 最後 世界 一切 一起 不想 如果 可以 不再 時候 一天 來 去 自己 為 一個 什麼 只是 還是 不是 不會 怎麼 知道 這個 就是 
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)
dictionary.compactify()
dictionary.save('lyrics.dict')

texts = [[word for word in document.split() if word not in stoplist]
         for document in open(input_train_data_file)]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

## Add Criteria: frequency < 1000
texts = [[token for token in text if frequency[token] > 1 and frequency[token] < 1000]
         for text in texts]

## Add Criteria: the word should contain 2 or more characters
texts = [[token for token in text if len(token) > 3]
         for text in texts]

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('lyrics.mm', corpus)

####################

## Explore frequency
short_frequency = defaultdict(int)
for token in frequency:
    if frequency[token] > 1 and token not in stoplist and len(token) > 3:
        short_frequency[token] = frequency[token]
sorted_freq = sorted(short_frequency.items(), key=operator.itemgetter(1), reverse=True)

n = 0
for item in sorted_freq[1:100]:
    n += 1
    print("{}. {} : {} 次".format(n, item[0], item[1]))

####################

if (os.path.exists("lyrics.dict")):
    dictionary = corpora.Dictionary.load('lyrics.dict')
    corpus = corpora.MmCorpus('lyrics.mm')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
corpus_lsi = lsi[corpus_tfidf]

## Try Lda model
#lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=5)
#lda.print_topics(2)

#lsi.save('lyrics.lsi')
#lsi = models.LsiModel.load('lyrics.lsi')