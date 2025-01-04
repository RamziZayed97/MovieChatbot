from flask import Flask, request, jsonify, render_template
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Initialize the Flask application
app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Sample corpus for the chatbot to work with
with open('moviedata.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenize sentences
sent_tokens = nltk.sent_tokenize(raw)

# Preprocessing function
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting function
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Response function
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.remove(user_response)
    return robo_response

# Flask route to serve HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Serve the chatbot HTML interface

# Flask route to handle chatbot responses
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']
    if greeting(user_input) is not None:
        return jsonify({'response': greeting(user_input)})
    else:
        return jsonify({'response': response(user_input)})

if __name__ == "__main__":
    app.run(debug=True)
