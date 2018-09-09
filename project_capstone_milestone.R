## load the data
library(NLP)
library(tm)
library(wordcloud)
library(RWeka)
library(SnowballC)
library(ggplot2)
library(rtweet)

US_blogs=readLines("C:\\Users\\frozenl\\Desktop\\coursera\\project_capstone\\Coursera_SwiftKey\\en_US\\en_US.blogs.txt", encoding = "UTF-8")
US_twitter=readLines("C:\\Users\\frozenl\\Desktop\\coursera\\project_capstone\\Coursera_SwiftKey\\en_US\\en_US.twitter.txt", encoding = "UTF-8")
US_news=readLines("C:\\Users\\frozenl\\Desktop\\coursera\\project_capstone\\Coursera_SwiftKey\\en_US\\en_US.news.txt", encoding = "UTF-8")

## preprocess the data
blogs_length=length(US_blogs)
twitter_length=length(US_twitter)
news_length=length(US_news)

blogs_size=file.info("C:\\Users\\frozenl\\Desktop\\coursera\\project_capstone\\Coursera_SwiftKey\\en_US\\en_US.blogs.txt")$size/1024^2
twitter_size=file.info("C:\\Users\\frozenl\\Desktop\\coursera\\project_capstone\\Coursera_SwiftKey\\en_US\\en_US.twitter.txt")$size/1024^2
news_size=file.info("C:\\Users\\frozenl\\Desktop\\coursera\\project_capstone\\Coursera_SwiftKey\\en_US\\en_US.news.txt")$size/1024^2

blogs_nchar=sum(nchar(US_blogs))
twitter_nchar=sum(nchar(US_twitter))
news_nchar=sum(nchar(US_news))

blogs_words=sum(sapply(strsplit(US_blogs, "\\s+"), length))
twitter_words=sum(sapply(strsplit(US_twitter, "\\s+"), length))
news_words=sum(sapply(strsplit(US_news, "\\s+"), length))

text_summary=data.frame(name=c("blogs","twitter","news"),
                        length=c(blogs_length,twitter_length,news_length),
                        size.MB=c(blogs_size,twitter_size,news_size),
                        character=c(blogs_nchar,twitter_nchar,news_nchar),
                        words=c(blogs_words,twitter_words,news_words))
text_summary

## data cleaning (sample size considering the memory of my laptop)
set.seed(7777)
sample_blogs=sample(US_blogs, 6000, replace=FALSE)
sample_twitter=sample(US_twitter, 6000, replace=FALSE)
sample_news=sample(US_news, 6000, replace=FALSE)


rm(US_blogs)
rm(US_twitter)
rm(US_news)


## build the corpus (clean_tweets can remove conjunctions)
corpus=function(sample){
    sample1=VCorpus(VectorSource(sample))
    sample2=tm_map(sample1, removeNumbers)
    sample3=tm_map(sample2, removePunctuation)
    sample4=tm_map(sample3, removeWords, stopwords("english"))
    sample5=tm_map(sample4, stripWhitespace)
    sample6=tm_map(sample5, stemDocument)
    sample7=tm_map(sample6, content_transformer(tolower))
    return(sample7)
}

sample_twitter=iconv(sample_twitter, "UTF-8", "ASCII")
blogs=corpus(clean_tweets(sample_blogs))
twitter=corpus(clean_tweets(sample_twitter))
news=corpus(clean_tweets(sample_news))

## summary the corpus 

distribution=function(text){
    tdm=TermDocumentMatrix(text)
    count=rowSums(as.matrix(tdm))
    print(head(sort(count, decreasing = T)))
    print(head(table(count), 10))
    wordcloud(text, max.words = 100, random.order = FALSE,rot.per=0.37, 
              use.r.layout=FALSE,colors=brewer.pal(8, "Set1"))
    
}

distribution(blogs)
distribution(twitter)
distribution(news)

## build n_gram models and show the plots
n_gram=function(n,sample_deal){
    m=function(x) NGramTokenizer(x, Weka_control(min=n, max=n))
    m_table=TermDocumentMatrix(sample_deal, control=list(tokenize=m))
    m_corpus=findFreqTerms(m_table)
    m_corpus_num=rowSums(as.matrix(m_table[m_corpus,]))
    m_corpus_table=data.frame(words=names(m_corpus_num),frequency=m_corpus_num)
    m_corpus_sort=m_corpus_table[order(-m_corpus_table$frequency),]
    result=m_corpus_sort[1:10,]
    print(result)
    n_g=ggplot(result,aes(x=reorder(words,-frequency),y=frequency,fill=factor(reorder(words,-frequency))))
    n_g=n_g+geom_bar(stat="identity")
    n_g=n_g+labs(title="n-gram",x="Words",y="Frequency",fill="Frequency")
    n_g=n_g+theme(axis.text.x=element_text(angle=90))
    return(n_g)
}

bi_blogs=corpus(sample(sample_blogs, 2000, replace=FALSE))
bi_twitter=corpus(sample(sample_twitter, 2000, replace=FALSE))
bi_news=corpus(sample(sample_news, 2000, replace=FALSE))

tri_blogs=corpus(sample(sample_blogs, 2000, replace=FALSE))
tri_twitter=corpus(sample(sample_twitter, 2000, replace=FALSE))
tri_news=corpus(sample(sample_news, 2000, replace=FALSE))

n_gram(1,blogs)
n_gram(2,bi_blogs)
n_gram(3,tri_blogs)
n_gram(1,twitter)
n_gram(2,bi_twitter)
n_gram(3,tri_twitter)
n_gram(1,news)
n_gram(2,bi_news)
n_gram(3,tri_news)

