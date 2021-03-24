# Hotel Review Classification - NLP
# Importing necessary modules
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.util import ngrams
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag 
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

# Importing train data
train = pd.read_csv("train.csv")
train.shape #(14343,3)
train.dtypes
train.isnull().sum() # No null values
train.head()

# Value counts of 'Ratings'
# Range of 'Rating' (1-5)
train['Rating'].unique()
train['Rating'].value_counts()
train['Rating'].value_counts()/14343 * 100
train['Rating'].describe()
train['Review'].nunique() #14343 unique reviews, no duplicate reviews
 
# Bar plot ratings vs no. of reviews
rating_count = train.groupby('Rating').count()
plt.bar(rating_count.index.values, rating_count['Review']);
plt.xlabel('Ratings');plt.ylabel('Reviews of hotel');plt.title('Frequency of Reviews');plt.show()

### Data cleaning and pre-processing ###
        
# Expanding all the contraction words
# Defining a dictionary of contractions
      
con = { "ain't": "are not","'s":" is","aren't": "are not", "can't": "cannot","can't've": "cannot have",
       "'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have",
       "didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have",
       "hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
       "he'll've": "he will have","how'd": "how did","how'd'y": "how do you","how'll": "how will","I'd": "I would",
       "I'd've": "I would have","I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have",
       "isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will","it'll've": "it will have",
       "let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not", 
       "mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have",
       "needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
       "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have",
       "she'd": "she would","she'd've": "she would have","she'll": "she will", "she'll've": "she will have",
       "should've": "should have","shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
       "that'd": "that would","that'd've": "that would have", "there'd": "there would","there'd've": "there would have",
       "they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have",
       "they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we would",
       "we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
       "weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
       "what've": "what have","when've": "when have","where'd": "where did", "where've": "where have","who'll": "who will",
       "who'll've": "who will have","who've": "who have","why've": "why have","will've": "will have","won't": "will not",
       "won't've": "will not have", "would've": "would have","wouldn't": "would not","wouldn't've": "would not have",
       "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",
       "y'all've": "you all have", "you'd": "you would","you'd've": "you would have","you'll": "you will",
       "you'll've": "you will have", "you're": "you are","you've": "you have", "n't":"not"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(con.keys()))

# Function for expanding contractions
def expand_contractions(text,contractions_dict=con):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, text)

# Expanding Contractions in the reviews
train['Review_con'] = train['Review'].apply(lambda x:expand_contractions(x))

pd.set_option('display.max_colwidth',500)# Increasing the max number of dispay words to 500

train['Review_con'].head()

# Removing punctuations, numbers, stopwords word tokenizing and lemmatizing  
stopwords = nltk.corpus.stopwords.words('english')
stopwords = set(stopwords)
print("Total count of all stopwords : " , len(stopwords)) # 179
print(stopwords)

lm = WordNetLemmatizer() # Importing necessary modules for text cleaning

# Defining a function for cleaning reviews
def clean_text(text):
    text = ''.join(char.lower() for char in text if char not in string.punctuation)
    text = re.sub('\w*\d\w*', '', text)
    word_tokens = re.split('\W+', text)
    text = [lm.lemmatize(word) for word in word_tokens if word not in stopwords]
    return text

# Cleaning reviews text in the dataframe
train['Cleaned_Review'] = train['Review_con'].apply(lambda x: clean_text(x))
train['Cleaned_Review'].head()

# Most frequently used words of cleaned reviews
text = train['Cleaned_Review'].apply(' '.join)  # to series
text = ''.join(text) # to string
text_new = text.split() # to list
len(text_new) # 1397581

# We observe some similar words and insignificant words 
# Converting tokenized 'CLeaned Review' to string form
train['Cleaned_str'] = train['Cleaned_Review'].apply(' '.join)

# Replacing about 70 similar words to their stem form
# Defining a dictionary of similar words and stem form

sim = {'stayed':'stay', 'lovely':'love', 'loved':'love', 'liked':'like', 'exceptionally':'exceptional', 'returned':'return', 
'complained':'complain', 'waited':'wait', 'complaint':'complain', 'requested':'request', 'walked':'walk', 'smelled':'smell', 
'located':'location', 'staying':'stay', 'enjoyed':'enjoy', 'arrived':'arrive', 'walking':'walk', 'tried':'try', 
'trying':'try', 'paid':'pay', 'paying':'pay', 'charged':'charge', 'leaving':'leave', 'disappointing':'disappoint', 
'disappointed':'disappoint', 'stained':'stain', 'refused':'refuse', 'noisy':'noise', 'poorly':'poor', 'informed':'inform', 
'helpful':'help', 'cleaning':'clean', 'cleaned':'clean', 'checking':'check', 'ended':'end', 'priced':'price', 'nicer':'nice', 
'offered':'offer', 'arrival':'arrive', 'served':'serve', 'expected':'expect', 'nicely':'nice', 'waiting':'wait', 'impressed':'impress', 
'crowded':'crowd', 'quickly':'quick', 'working':'work', 'worked':'work', 'surprised':'surprise', 'provided':'provide', 
'reading':'read', 'including':'include', 'included':'include', 'relaxing':'relax', 'nyc':'nice', 'recommendation':'recommend', 'greeted':'greet', 
'thanks':'thank', 'appointed':'appoint', 'upgraded':'upgrade', 'decorated':'decor','swimming':'swim' }

# Regular expression for finding similar words
sim_re=re.compile('(%s)' % '|'.join(sim.keys()))

# Function for expanding contractions
def stemming(text, sim_dict= sim):
  def replace(match):
    return sim_dict[match.group(0)]
  return sim_re.sub(replace, text)

# Converting similar words to their stem form in the reviews
train['Cleaned_str'] = train['Cleaned_str'].apply(lambda x:stemming(x))
train['Cleaned_str'].head()

# Removing all insignificant words found in the frequently used 500 words
insig =['didn__ç_é_','gone','wo','despite','sent','pm','knew','yes','one','w','got',
          'told', 'maybe','going','want','need','took','went','absolutely','immediately',
          'coming','advised','getting','sure','gave','asked','n','e,','generally','didnt',
          'tell','ok','given','ask','looked','let','wanted','came','finally','say','said'
         'oh','saying','c','definitely','probably','especially','dont','really','called'
         ,'usually','certainly','unless','decided','included','include','left','kept',
         'needed','actually','looked','called','given','used','ask','getting','know',
         'took','make','really','taken','okay','u','chose','st','making','seen','brought'
         'bring','look','wanted','come','looking','go','tell','saw','definately','asked'
         ,'getting','know','taking','set','okay','saw','later','nt','...','didnçé','hotel','room',
         'resort','restaurant']

# Forming a variable 'pat' to pass reviews to remove insignificant words
pat = r'\b(?:{})\b'.format('|'.join(insig))

train['Cleaned_str'] = train['Cleaned_str'].str.replace(pat, '')
train['Cleaned_str']= train['Cleaned_str'].str.replace('  ',' ') # Removing whitespaces

# Converting to string
text_clean = ''.join(train['Cleaned_str'])
text_clean = text_clean.split() # to list form
len(text_clean) #1117278

# Parts of speech (POS) tagging
train['Tagged_words'] = train['Cleaned_str'].map(pos_tag)
train['Tagged_words'].head()

# Creating a function for counts of tagged words
def count_tags(title_with_tags):
    tag_count = {}
    for word, tag in title_with_tags:
        if tag in tag_count:
            tag_count[tag] += 1
        else:
            tag_count[tag] = 1
    return(tag_count)

train['Tag_counts'] = train['Tagged_words'].map(count_tags)
train['Tag_counts'].head()

# Function for extracting only 'Noun' words
def find_noun(keyword):
    tokens = nltk.word_tokenize(keyword)
    tagged = nltk.pos_tag(tokens)
    noun = [w for w,t in tagged if 'NN' in t] 
    return noun
    
# Function for extracting only 'Adjective' words

def find_adj(keyword):
    tokens = nltk.word_tokenize(keyword)
    tagged = nltk.pos_tag(tokens)
    adj = [w for w,t in tagged if 'JJ' in t] 
    return adj

# Extracting only 'noun' words
train['Noun'] = train['Cleaned_str'].apply(find_noun)
text_noun = train['Noun'].apply(' '.join)
text_noun = ' '.join(text_noun)
text_noun = text_noun.split() # to list form
len(text_noun) # 710916

# Frequency of commonly used noun words
Freq_words = FreqDist(text_noun)
Freq_words.most_common(60)
Freq_words.plot(30)

# Extracting only 'adjectives' words
train['Adjective'] = train['Cleaned_str'].apply(find_adj)
text_adj = train['Adjective'].apply(' '.join)
text_adj = ' '.join(text_adj)
text_adj = text_adj.split() # to list form
len(text_adj) # 349119

# Frequency of commonly used adjectives words
Freq_words = FreqDist(text_adj)
Freq_words.most_common(60)
Freq_words.plot(30)

# Most frequently used words
Freq_words = FreqDist(text_clean)
Freq_words.most_common(30)
Freq_words.plot(30)

# Converting list to string
# Creating function for it
def listToString(s):  
    str1 = " " 
    return (str1.join(s)) 
cleantext_wordcloud = listToString(text_clean)

# Forming a generic wordcloud of most frequent words
wc=WordCloud(max_words =100,min_word_length=2,background_color='black',width=2200,height=1400)
word_cloud = wc.generate(cleantext_wordcloud)
plt.figure(figsize = (12,12));plt.imshow(word_cloud);plt.axis("off");plt.show()

# Sentiment scores
cln_train_str = train['Cleaned_str']

bloblist_tags = list()

for row in cln_train_str:
    blob = TextBlob(row)
    bloblist_tags.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    train_polarity = pd.DataFrame(bloblist_tags, columns = ['sentence','polarity','subjectivity'])

# Merging 'train_polarity' dataframe to 'train' dataframe creating a temporary column to inner join
train['tmp'] = range(0,14343,1)
train_polarity['tmp'] = range(0,14343,1)

train = pd.merge(train, train_polarity,how ='inner', on=['tmp'])
train = train.drop(['ID', 'tmp', 'sentence'], axis = 1)

# Sentiment type of review based on polarity scores (Positive, neutral or negative)
def f(train):
    if train['polarity'] > 0:
        val = "Positive"
    elif train['polarity'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

train['Sentiment_Type'] = train.apply(f, axis=1)
train['Sentiment_Type'].head(10)

# Bar plot of sentiment type
sns.set_style("whitegrid");plt.title("Sentiment distribution of Reviews");plt.figure(figsize=(50,50));
ax = sns.countplot(x="Sentiment_Type", data=train)

# Positive and negative wordcloud
# Wc on first 10000 words as sample
cleantext_sample = cleantext_wordcloud[0:100000]
len(cleantext_sample)

# Loading positive set of words for generating positive word cloud
with open("positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]

# Loading Negative set of words for generating positive word cloud
with open("negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# Negative word cloud
# Choosing only the words which are present in negwords
cleantxt_neg = "".join(cleantext_sample) # Converting to string
cleantxt_neg_list = cleantxt_neg.split() # Separating words by comma
neg_words_review = [w for w in cleantxt_neg_list if w in negwords]
neg_words_str = " ".join(neg_words_review)

wordcloud_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(neg_words_str)

plt.imshow(wordcloud_neg);plt.axis("off");plt.title("Negative WordCloud")

#Frequency plot of most commonly used negative words
Frq_words = FreqDist(neg_words_review)
Frq_words.most_common(30)
Frq_words.plot(30)

# Positive word cloud
# Choosing the only words which are present in positive words
cleantxt_pos = "".join(cleantext_sample)
cleantxt_pos_list = cleantxt_pos.split()
pos_words_review = [w for w in cleantxt_pos_list if w in poswords]
pos_words_str = " ".join(pos_words_review)

wordcloud_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(pos_words_str)

plt.imshow(wordcloud_pos);plt.axis("off");plt.title("Positive WordCloud")

#Frequency plot of most commonly used positive words
Frq_words = FreqDist(pos_words_review)
Frq_words.most_common(30)
Frq_words.plot(30)

# Sentiment scores as a feature
# Creating a dataframe of mean scores of polarity and subjectivity 
# Sentiment scores on each rating
txt_sent_score_5 = train[train['Rating'] == 5][['polarity','subjectivity']]
txt_sent_score_4 = train[train['Rating'] == 4][['polarity','subjectivity']]
txt_sent_score_3 = train[train['Rating'] == 3][['polarity','subjectivity']]
txt_sent_score_2 = train[train['Rating'] == 2][['polarity','subjectivity']]
txt_sent_score_1 = train[train['Rating'] == 1][['polarity','subjectivity']]

p1 = txt_sent_score_1['polarity'].mean()
p2 = txt_sent_score_2['polarity'].mean()
p3 = txt_sent_score_3['polarity'].mean()
p4 = txt_sent_score_4['polarity'].mean()
p5 = txt_sent_score_5['polarity'].mean()

s1 = txt_sent_score_1['subjectivity'].mean()
s2 = txt_sent_score_2['subjectivity'].mean()
s3 = txt_sent_score_3['subjectivity'].mean()
s4 = txt_sent_score_4['subjectivity'].mean()
s5 = txt_sent_score_5['subjectivity'].mean()

lst =  [[1,p1,s1],[2,p2,s2],[3,p3,s3],[4,p4,s4],[5,p5,s5]]

sent_score_mean = pd.DataFrame(lst, columns=['Rating','Polarity','Subjectivity'])
sent_score_mean.head()

# We see a variation in the polarity mean scores for all ratings
# Plotting histogram for polarity scores of all ratings
bins = np.linspace(-1,1,200)

plt.hist(txt_sent_score_5['polarity'], bins, label = '5 ♣');
plt.hist(txt_sent_score_4['polarity'], bins, label = '4 ♣');
plt.hist(txt_sent_score_3['polarity'], bins, label = '3 ♣');
plt.hist(txt_sent_score_2['polarity'], bins, label = '2 ♣');
plt.hist(txt_sent_score_1['polarity'], bins, label = '1 ♣');
plt.legend(loc = 'upper right');
plt.xlabel('Polarity');
plt.ylabel('Frequency of Reviews');
plt.title('Polarity scores over ratings');
plt.show()

# Feature extraction using N-Gram for each rating
# 1 Rating
text1 = train[train['Rating'] == 1]['Cleaned_str']
text1 = ''.join(text1)
text1 = text1.split()

# N - grams @ rating 1
# Bigram
bi1grams = ngrams(text1, 2)
bi1gramFreq = collections.Counter(bi1grams)

Frq_words1b = FreqDist(bi1gramFreq)
Frq_words1b.most_common(40)
Frq_words1b.plot(40) 

# Trigram
t1grams = ngrams(text1, 3)
t1gramFreq = collections.Counter(t1grams)

Frq_word1t = FreqDist(t1gramFreq)
Frq_word1t.most_common(40)
Frq_word1t.plot(40)

# Quadgram
q1grams = ngrams(text1, 4)
q1gramFreq = collections.Counter(q1grams)

Frq_word1q = FreqDist(q1gramFreq)
Frq_word1q.most_common(40)
Frq_word1q.plot(40)

# Frequency of most used words in rating 1
Freq_wc1 = FreqDist(text1)
Freq_wc1.most_common(40)
Freq_wc1.plot(40)

# 2 Rating
text2 = train[train['Rating'] == 2]['Cleaned_str']
text2 = ''.join(text2)
text2 = text2.split()

# N - grams @ rating 2
# Bigram
bi2grams = ngrams(text2, 2)
bi2gramFreq = collections.Counter(bi2grams)

Frq_words2b = FreqDist(bi2gramFreq)
Frq_words2b.most_common(40)
Frq_words2b.plot(40)

# Trigram
t2grams = ngrams(text2, 3)
t2gramFreq = collections.Counter(t2grams)

Frq_word2t = FreqDist(t2gramFreq)
Frq_word2t.most_common(40)
Frq_word2t.plot(40)

# Quadgram
q2grams = ngrams(text2, 4)
q2gramFreq = collections.Counter(q2grams)

Frq_word2q = FreqDist(q2gramFreq)
Frq_word2q.most_common(40)
Frq_word2q.plot(40)

# Frequency of most used words in rating 2
Freq_wc2 = FreqDist(text2)
Freq_wc2.most_common(40)
Freq_wc2.plot(40)

# 3 Rating
text3 = train[train['Rating'] == 3]['Cleaned_str']
text3 = ''.join(text3)
text3 = text3.split()

# N - grams @ rating 3
# Bigram
bi3grams = ngrams(text3, 2)
bi3gramFreq = collections.Counter(bi3grams)

Frq_words3b = FreqDist(bi3gramFreq)
Frq_words3b.most_common(40)
Frq_words3b.plot(40)

# Trigram
t3grams = ngrams(text3, 3)
t3gramFreq = collections.Counter(t3grams)

Frq_word3t = FreqDist(t3gramFreq)
Frq_word3t.most_common(40)
Frq_word3t.plot(40)

# Quadgram
q3grams = ngrams(text3, 4)
q3gramFreq = collections.Counter(q3grams)

Frq_word3q = FreqDist(q3gramFreq)
Frq_word3q.most_common(40)
Frq_word3q.plot(40)

# Frequency of most used words in rating 3
Freq_wc3 = FreqDist(text3)
Freq_wc3.most_common(40)
Freq_wc3.plot(40)

# 4 Rating
text4 = train[train['Rating'] == 4]['Cleaned_str']
text4 = ''.join(text4)
text4 = text4.split()

# N - grams @ rating 4
# Bigram
bi4grams = ngrams(text4, 2)
bi4gramFreq = collections.Counter(bi4grams)

Frq_words4b = FreqDist(bi4gramFreq)
Frq_words4b.most_common(40)
Frq_words4b.plot(40)

# Trigram
t4grams = ngrams(text4, 3)
t4gramFreq = collections.Counter(t4grams)

Frq_word4t = FreqDist(t4gramFreq)
Frq_word4t.most_common(40)
Frq_word4t.plot(40)

# Quadgram
q4grams = ngrams(text4, 4)
q4gramFreq = collections.Counter(q4grams)

Frq_word4q = FreqDist(q4gramFreq)
Frq_word4q.most_common(40)
Frq_word4q.plot(40)

# Frequency of most used words in rating 4
Freq_wc4 = FreqDist(text4)
Freq_wc4.most_common(40)
Freq_wc4.plot(40)

# 5 Rating
text5 = train[train['Rating'] == 5]['Cleaned_str']
text5 = ''.join(text5)
text5 = text5.split()

# N - grams @ rating 5
# Bigram
bi5grams = ngrams(text5, 2)
bi5gramFreq = collections.Counter(bi5grams)

Frq_words5b = FreqDist(bi5gramFreq)
Frq_words5b.most_common(40)
Frq_words5b.plot(40)

# Trigram
t5grams = ngrams(text5, 3)
t5gramFreq = collections.Counter(t5grams)

Frq_word5t = FreqDist(t5gramFreq)
Frq_word5t.most_common(40)
Frq_word5t.plot(40)

# Quadgram
q5grams = ngrams(text5, 4)
q5gramFreq = collections.Counter(q5grams)

Frq_word5q = FreqDist(q5gramFreq)
Frq_word5q.most_common(40)
Frq_word5q.plot(40)

# Review length as a feature
# Featuring the lenghts of each review
# Creating a new column for lenghts of each review
train['Review_len']= train['Cleaned_Review'].apply(lambda x: len(x))
max(train['Review_len']) #1848
min(train['Review_len']) #9
train.columns
train['Review_len'].describe()

# Review length for each rating
text_review_5 = train[train['Rating'] == 5]['Review_len']
text_review_4 = train[train['Rating'] == 4]['Review_len']
text_review_3 = train[train['Rating'] == 3]['Review_len']
text_review_2 = train[train['Rating'] == 2]['Review_len']
text_review_1 = train[train['Rating'] == 1]['Review_len']

text_review_5.describe()
text_review_4.describe()
text_review_3.describe()
text_review_2.describe()
text_review_1.describe()

# Plotting histogram of 'Review_len' column
bins = np.linspace(0,400,50)
plt.hist(train['Review_len'], bins);
plt.xlabel('Length of Review');plt.ylabel('Number of reviews');
plt.title("Frequency of lengths");
plt.show()

# The plot is slightly positively skewed

# Boxplot of 'Review_Length'
plt.boxplot(train['Review_len'])
# We see a lot of outliers, so restricting lenght of reviews to 500

train['Review_len_500'] = train['Review_len'] < 500
train['less_500'] = train[train['Review_len_500'] == True]['Review_len']

bins = np.linspace(0,400,100)
plt.hist(train['less_500'], bins) # Histogram of review length less than 500

# Out of all transfroms, root 6 transform looks normally distributed
# Transforming 'Review Length' by root 6 transform
train['Transformed_rev_leng'] = train['Review_len']**(1/6)

bins = np.linspace(0,50,50)
max(train['Transformed_rev_leng']) #3.5
min(train['Transformed_rev_leng']) #1.44
plt.hist(train['Transformed_rev_leng'], bins= 50)
train['Transformed_rev_leng'].describe()

# Frequency of 'Trasnformed length reviews' w.r.t each rating
bins = np.linspace(0,5,100)
plt.hist(train[train['Rating'] == 5]['Transformed_rev_leng'], bins, label = '5 ♣');
plt.hist(train[train['Rating'] == 4]['Transformed_rev_leng'], bins, label = '4 ♣');
plt.hist(train[train['Rating'] == 3]['Transformed_rev_leng'], bins, label = '3 ♣');
plt.hist(train[train['Rating'] == 2]['Transformed_rev_leng'], bins, label = '2 ♣');
plt.hist(train[train['Rating'] == 1]['Transformed_rev_leng'], bins, label = '1 ♣');
plt.legend(loc = 'upper right');
plt.xlabel('Transformed length of a Review');
plt.ylabel('Frequency of Reviews')
plt.title('Frequency of varied lenghts of Reviews');
plt.show()

# Kurtosis is the only parameter which is unique in histagrams of all 5 ratings
from scipy.stats import kurtosis
kurtosis(train[train['Rating'] == 5]['Transformed_rev_leng']) #1.38
kurtosis(train[train['Rating'] == 4]['Transformed_rev_leng']) #1.15
kurtosis(train[train['Rating'] == 3]['Transformed_rev_leng']) #1.13
kurtosis(train[train['Rating'] == 2]['Transformed_rev_leng']) #0.72
kurtosis(train[train['Rating'] == 1]['Transformed_rev_leng']) #0.49

# Correlation between all featured variables
corr_matrix = train.corr().abs() # Correlation Matrix
corr = train.corr()
corr

# Heat map of correlation
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

# Model building
# Importing 'test' dataset
test = pd.read_csv("test.csv")
test['Cleaned_Review'] = test['Review'].apply(lambda x: clean_text(x))
test['Cleaned_str'] = test['Cleaned_Review'].apply(', '.join)
test['Review_len'] = test['Cleaned_Review'].apply(lambda x: len(x))
test['Transformed_rev_leng'] = test['Review_len']**(1/6)

# Sentiment scores of 'test' data
bloblist_tags = list()

txt_sent_score_ = test['Cleaned_Review']
cln_test_str = txt_sent_score_.astype(str)

for row in cln_test_str:
    blob = TextBlob(row)
    bloblist_tags.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    test_polarity = pd.DataFrame(bloblist_tags, columns = ['sentence','polarity','subjectivity'])

# Merging 'Polarity" dataframe with 'train'
test['tmp'] = range(0,6148,1)
test_polarity['tmp'] = range(0,6148,1)

test = pd.merge(test, test_polarity,how ='inner', on=['tmp'])
test.columns

# Importing 'Sample submission' dataset
test_y = pd.read_csv("sample_submission.csv")

# Model building on different features using various classifiers

# Train models
def svc_tr(x,y):
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.3, random_state = 0)
    classSVC = LinearSVC(class_weight= 'balanced') #the class_weight="balanced" option tries to remove the biasedness of model towards majority sample
    classSVC.fit(x_train, y_train)
    pred_svc = classSVC.predict(x_test)
    print("Linear SVC:",accuracy_score(y_test, pred_svc))
   
def knn_tr(x,y):
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.3, random_state = 0, stratify = y)
    classKNN = KNeighborsClassifier(n_neighbors=5)
    classKNN.fit(x_train, y_train)
    pred_knn = classKNN.predict(x_test)
    print("kNN:",accuracy_score(y_test, pred_knn))

def nb_tr(x,y):
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.3, random_state = 0, stratify = y)
    classNB = MultinomialNB()
    classNB.fit(x_train, y_train)
    pred_nb = classNB.predict(x_test)
    print("Naive Bayes:",accuracy_score(y_test, pred_nb))

def rf_tr(x,y):
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.3, random_state = 0, stratify = y)
    classRF = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    classRF.fit(x_train, y_train)
    pred_rf = classRF.predict(x_test)
    print ('accracy is: ', accuracy_score(y_test, pred_rf))

# Test models    
def svc_t(x_train, x_test, y_train, y_test):
    classSVC = LinearSVC(class_weight= 'balanced')
    classSVC.fit(x_train, y_train)
    global pred_svc
    pred_svc = classSVC.predict(x_test)
    print("Linear SVC:",accuracy_score(y_test, pred_svc))
   
def knn_t(x_train, x_test, y_train, y_test):
    classKNN = KNeighborsClassifier(n_neighbors=5)
    classKNN.fit(x_train, y_train)
    global pred_knn
    pred_knn = classKNN.predict(x_test)
    print("kNN:",accuracy_score(y_test, pred_knn))

def nb_t(x_train, x_test, y_train, y_test):
    classNB = MultinomialNB()
    classNB.fit(x_train, y_train)
    global pred_nb
    pred_nb = classNB.predict(x_test)
    print("Naive Bayes:",accuracy_score(y_test, pred_nb))

def rf_t(x_train, x_test, y_train, y_test):
    classRF = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    classRF.fit(x_train, y_train)
    global pred_rf
    pred_rf = classRF.predict(x_test)
    print ('accracy is: ', accuracy_score(y_test, pred_rf))
    
### MODEL BUILDING on '3' Ratings (1 = least, 2 = neutral, 3 = best) ###

# Converting train '5' Ratings to '3' Ratings
train['Rating'] = train['Rating'].replace(2,1)
train['Rating'] = train['Rating'].replace(3,2)
train['Rating'] = train['Rating'].replace(4,2)
train['Rating'] = train['Rating'].replace(5,3)
train['Rating'].value_counts()
train['Rating'].unique()

# Converting sample_submission '5' Ratings to '3' Ratings
test_y['Rating'] = test_y['Rating'].replace(2,1)
test_y['Rating'] = test_y['Rating'].replace(3,2)
test_y['Rating'] = test_y['Rating'].replace(4,2)
test_y['Rating'] = test_y['Rating'].replace(5,3)
test_y['Rating'].value_counts()
test_y['Rating'].unique()

test_y = test_y['Rating']

### Model building on transformed lenght of reviews ###
train_xl = train['Transformed_rev_leng']
train_y = train['Rating']

train_xl = train_xl.to_numpy()
train_y = train_y.to_numpy()

train_xl = train_xl.reshape(-1, 1)
train_y = train_y.reshape(-1, 1)

train_y = train_y.ravel()

oversample = SMOTE()
train_xl, train_yl = oversample.fit_resample(train_xl, train_y)

# Train accuracy
svc_tr(train_xl,train_yl) # 37.35
knn_tr(train_xl,train_yl) # 35.54
nb_tr(train_xl,train_yl) # 33.33
rf_tr(train_xl,train_yl) # 38.58

# Test accuracy 
test_xl = test['Transformed_rev_leng']
test_xl = test_xl.to_numpy()
test_xl = test_xl.reshape(-1,1)

svc_t(train_xl,test_xl,train_yl,test_y) # 33.75
knn_t(train_xl,test_xl,train_yl,test_y) # 30.54
nb_t(train_xl,test_xl,train_yl,test_y) # 15.92
rf_t(train_xl,test_xl,train_yl,test_y) # 36.40

### Model building on polarity scores ###
train_xp = train['polarity']
train_y = train['Rating']

train_xp = train_xp.to_numpy()
train_y = train_y.to_numpy()

train_xp = train_xp.reshape(-1, 1)
train_y = train_y.reshape(-1, 1)

train_y = train_y.ravel()

oversample = SMOTE()
train_xp, train_yp = oversample.fit_resample(train_xp, train_y)

# Train accuracy 
svc_tr(train_xp,train_yp) # 57.34
knn_tr(train_xp,train_yp) # 57.34
nb_tr(train_xp,train_yp) # -
rf_tr(train_xp,train_yp) # 54.56

# Test accuracy 
test_xp = test_polarity['polarity']
test_xp = test_xp.to_numpy()
test_xp = test_xp.reshape(-1, 1)

svc_t(train_xp,test_xp,train_yp,test_y) # 49.56
knn_t(train_xp,test_xp,train_yp,test_y) # 48.23
nb_t(train_xp,test_xp,train_yp,test_y)
rf_t(train_xp,test_xp,train_yp,test_y) # 47.38

### Model building on raw reviews ###
vect = TfidfVectorizer(ngram_range = (1,5))
train_xr = train['Review']
train_y = train['Rating']
train_x_vect_r = vect.fit_transform(train_xr)

oversample = SMOTE()
train_x_vect_r, train_yr = oversample.fit_sample(train_x_vect_r, train_y)

# Train accuracy 
svc_tr(train_x_vect_r,train_yr) # 83.43
knn_tr(train_x_vect_r,train_yr) # 43.58
nb_tr(train_x_vect_r,train_yr) # 78.06
rf_tr(train_x_vect_r,train_yr) # 76.38

# Test accuracy 
test_xr = test['Review']
test_x_vect_r = vect.transform(test_xr)

svc_t(train_x_vect_r,test_x_vect_r,train_yr,test_y) # 73.05
knn_t(train_x_vect_r,test_x_vect_r,train_yr,test_y) # 28.30
nb_t(train_x_vect_r,test_x_vect_r,train_yr,test_y) # 65.14
rf_t(train_x_vect_r,test_x_vect_r,train_yr,test_y) # 62.05


### Model building on cleaned reviews ###
vect2 = TfidfVectorizer(ngram_range = (1,5))
train_xcr = train['Cleaned_str']
train_x_vect_cr = vect2.fit_transform(train_xcr)
train_y = train['Rating']

oversample = SMOTE()
train_x_vect_cr, train_cr = oversample.fit_resample(train_x_vect_cr, train_y)

# Train accuracy 
svc_tr(train_x_vect_cr,train_cr) # 82.78
knn_tr(train_x_vect_cr,train_cr) # 39.84
nb_tr(train_x_vect_cr,train_cr) # 79.54
rf_tr(train_x_vect_cr,train_cr) # 76.24

# Test accuracy 
test_xr = test['Cleaned_str']
test_x_vect_cr = vect2.transform(test_xr)

svc_t(train_x_vect_cr,test_x_vect_cr,train_cr,test_y) # 70.83
knn_t(train_x_vect_cr,test_x_vect_cr,train_cr,test_y) # 25.57
nb_t(train_x_vect_cr,test_x_vect_cr,train_cr,test_y) # 66.00
rf_t(train_x_vect_cr,test_x_vect_cr,train_cr,test_y) # 61.11

# Final Model
# We have best accuracy for the SVC model on TF-IDF of 'Review' = 73.05 % 
svc_t(train_x_vect_r,test_x_vect_r,train_yr,test_y) # 73.05 %  
confusion_matrix(test_y,pred_svc)
classification_report(test_y,pred_svc)
