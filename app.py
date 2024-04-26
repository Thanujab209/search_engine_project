from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer # type: ignore
import re
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
import chromaDB # type: ignore
from chromadb import ChromaDB # type: ignore

app = Flask(__name__)

# model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# ChromaDB 
client = chromadb.PersistentClient(path=r"D:\Innomatics\search_engine\work_0\chromadb") # type: ignore

collection = client.get_collection(name="collection1")

@app.route('/', methods=['GET', 'POST'])
def index():
        text = request.form['text']
        cleaned_text = preprocess_text(text)
        query_embedding = model.encode([cleaned_text])

        file1 = results.get('documents', []) # type: ignore
        metadata = results.get('metadatas', []) # type: ignore

        file1 = []
        file1.append({'document': doc, 'metadata': meta}) # type: ignore
        
        return text=text, documents=file1



def preprocess_text(raw_text, flag):
    # Removing special characters and digits
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', str(raw_text))
    sentence = re.sub("[^a-zA-Z]", " ", str(text))
    
    # change sentence to lower case
    sentence = sentence.lower()

    # tokenize into words
    tokens = sentence.split()
    
    # remove stop words                
    clean_tokens = [t for t in tokens if not t in stopwords.words("english")]
    
    # Stemming/Lemmatization
    if(flag == 'stem'):
        clean_tokens = [stemmer.stem(word) for word in clean_tokens] # type: ignore
    else:
        clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens] # type: ignore
    
    return pd.Series([" ".join(clean_tokens)]) # type: ignore

if __name__ == '__main__':
    app.run(debug=True)