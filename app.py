#from cgitb import text
#from distutils.log import debug
from flask import Flask
#from datetime import datetime
import re
from flask import render_template
from flask import Flask, request, render_template
# Import libraries
import pandas as pd
import nltk 
#from wordcloud import WordCloud, STOPWORDS , ImageColorGenerato
import string
import re
pd.set_option('display.max_colwidth', 100)
from gensim.corpora import Dictionary
from gensim import corpora, models
import gensim
from gensim import corpora, models, similarities
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from gensim.test.utils import common_texts
import collections
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#nltk.download('stopwords')

df = pd.DataFrame(columns=["hadith_search"])


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("searchengine4.html")
    
my_value = ""



@app.route('/', methods=['POST','GET'])
def my_form_post():

    text = request.form['u']
    processed_text = text.upper()
    user_hadith = str(processed_text)
    df = pd.DataFrame(data=[[user_hadith]],columns=["hadith_search"])
    df['text_enpunct'] = df['hadith_search'].apply(lambda x: remove_punct(x))
    df['hadith_tokenized'] = df['text_enpunct'].apply(lambda x: tokenization(x.lower()))
    df['hadith_nonstop'] = df['hadith_tokenized'].apply(lambda x: remove_stopwords(x))
    df['hadith_stemmed'] = df['hadith_nonstop'].apply(lambda x: stemming(x))
    df['hadith_lemmatized'] = df['hadith_stemmed'].apply(lambda x: lemmatizer(x))
    df['hadith_string'] = df['hadith_lemmatized'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
    all_words = ' '.join([word for word in df['hadith_string']])
    tokenized_words = nltk.tokenize.word_tokenize(all_words)
    fdist = FreqDist(tokenized_words)
    df['text_string_fdist'] = df['hadith_lemmatized'].apply(lambda x: ' '.join([item for item in x if fdist[item] >= 1 ]))
    # dropping the redundant colls
    df.drop('hadith_lemmatized', axis=1, inplace=True)
    df.drop('hadith_string', axis=1, inplace=True)
    df.drop('hadith_tokenized', axis=1, inplace=True)
    df.drop('hadith_nonstop', axis=1, inplace=True)
    df.drop('hadith_stemmed', axis=1, inplace=True)

    # what you could do is dropp the rest of the collumns save the text they put in. 
    # and use an insance of the df above to do the back end dev as not to play with collumn drops
    #https://discuss.dizzycoding.com/how-to-show-a-pandas-dataframe-into-a-existing-flask-html-table/ table intro
    df1= df.copy()
    df1.drop('text_enpunct', axis=1, inplace=True)
    tokens = str(df.at[0, 'text_string_fdist'])
    df1.drop('text_string_fdist', axis=1, inplace=True)
    df1.rename(columns={'hadith_search': 'The text you typed in!'}, inplace=True)
    #df 1 is soley for the display

    displaydf = most_similar(tokens)
     #NARRATED JABIR BIN `ABDULLAH: THE PROPHET USED TO POUR WATER THREE TIMES ON HIS HEAD.
    
    

  
    return render_template("searchengine4.html", processed_text = processed_text,column_names=df1.columns.values, row_data=list(df1.values.tolist()) ,texttype=displaydf.columns.values, results=list(displaydf.values.tolist()),    zip=zip)





# fucntions for cleaning 

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def tokenization(text):
    text = re.split('\W+', text)
    return text

stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

# functions for cleaning. above##################################################################################################

# this if you want local machine access
#df1 = pd.read_csv('/Users/ismailbhamjee/Documents/hadith nlp/notebooks/all_hadith_clean_df_2.csv', delimiter = ',')

df1 = pd.read_csv('all_hadith_clean_df_2.csv', delimiter = ',')

#df = df1[:500] # for ease of use 
df = df1




df.reset_index(drop=True, inplace=True)
train = df

#creates a tagged vector list
ls= gensim.models.doc2vec.TaggedDocument # this is a libary input not a c
train_corpus = []
j=0
for x in train['text_string_fdist'].values:
    
    train_corpus.append(ls(x.split(),[j]))
    j+=1
print('number of texts processed', j)

#model parameters
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40) #model parameters need updating
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

def most_similar(tokens):   
    tokens = tokens.split()
    ivec = model.infer_vector(doc_words=tokens, alpha=0.025) # ivec is inferfector 
    text = model.dv.most_similar(positive=[ivec], topn=10)    
    document_index_list= []
    english_text= []
    arabic_text = []
    book = []
    
    for x in text:
    
        y = str(x)
        y = y.replace("(", " #")
        y = y.replace(",", "$")
        y = y.split("$", 1)[0]
        y = y.split("#", 1)[1]
        y = y.strip(" ' ")
    
        document_index_list.append(y)


    for x in document_index_list:
        y = int(x)
        english_text.append(train.loc[y,'text_en'])
        arabic_text.append(train.loc[y,'text_ar'])
        book.append(train.loc[y,'source'])    
    displaydf = pd.DataFrame({'arabic':arabic_text})
    displaydf['english'] = english_text
    displaydf['book']= book
        
        
        
    return displaydf


if __name__ == '__main__':
    app.run(debug = True, host= '0,0,0,0')