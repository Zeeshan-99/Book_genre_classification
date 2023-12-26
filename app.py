
import re
import nltk
import pickle
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask import Flask, request, render_template

nltk.download('stopwords')
nltk.download('wordnet')

def preprocessing_text(text):
    text= re.sub("'\''","", text)
    text= re.sub("[^a-zA-Z]"," ", text)
    text= ' '.join(text.split())
    text= text.lower()
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words= text.lower()
    words = words.split()
    cleaned_text= [char for char in words if char not in stop_words]
    return ' '.join(cleaned_text)

def lematizing(sentence):
    lemma = WordNetLemmatizer()
    stemSentence = ""
    for word in sentence.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def Porter_stemming(text):
    stemmer = PorterStemmer()
    stemmed_sentences= ""
    for word in text.split():
        stemmed_word = stemmer.stem(word)
        stemmed_sentences += stemmed_word
        stemmed_sentences +=" "
    stemmed_sentences = stemmed_sentences.strip()
    return stemmed_sentences 


def test_function(text, model, tfidf_vectorizer):
    text= preprocessing_text(text)
    text= remove_stopwords(text)
    text= lematizing(text)
    text= Porter_stemming(text)
    
    text_vector= tfidf_vectorizer.transform([text])
    predicted= model.predict(text_vector)
    
    mapper = {0: 'Fantasy', 1: 'Science Fiction', 2: 'Crime Fiction',
                 3: 'Historical novel', 4: 'Horror', 5: 'Thriller'}

    return mapper[predicted[0]]
    

######------------------------------------------------
file = open('bookgenremodel.pkl','rb')
model = pickle.load(file)
file.close()

file1 = open('tfidfvectorizer.pkl','rb')
tfidf_vectorizer = pickle.load(file1)
file1.close()


app= Flask(__name__)

@app.route('/', methods=['GET','POST'])

def hello_world():
    if request.method == 'POST':
        mydict= request.form
        text = mydict["summary"]
        prediction = test_function(text, model, tfidf_vectorizer)
        return render_template('index.html', genre= prediction, text= str(text)[:100], showresult=True)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


