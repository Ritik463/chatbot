import random
import string
import nltk
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Reading in the corpus (chatbot.txt)
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Preprocessing
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Define intents
intents = {
    'greeting': {
        'patterns': ['hello', 'hi', 'hey', 'howdy'],
        'responses': ['Hi there!', 'Hello!', 'Hey!', 'Greetings!']
    },
    'developer': {
        'patterns': ['who developed you', 'who made you', 'creator'],
        'responses': ['Ritik is my developer.', 'I was developed by Ritik.']
    },
    'common_cold': {
        'patterns': ['common cold', 'cold symptoms', 'cold treatment'],
        'responses': [
            'The common cold is a viral infection of the upper respiratory tract.',
            'Symptoms of a common cold include runny nose, sore throat, and cough.',
            'There is no cure for the common cold, but symptoms can be managed with rest and fluids.'
        ]
    },
    'tata_technologies': {
        'patterns': ['tata technologies', 'what does tata technologies do', 'tata technologies services'],
        'responses': [
            'Tata Technologies provides engineering and IT services across various industries.',
            'They support product lifecycle management and digital manufacturing solutions.',
            'Tata Technologies operates globally with a focus on automotive, aerospace, and other sectors.'
        ]
    }
    # Add more intents as needed
}

# Generating response based on intent
def response(user_response):
    for intent, data in intents.items():
        for pattern in data['patterns']:
            if pattern in user_response.lower():
                return random.choice(data['responses'])

    # TF-IDF and cosine similarity based response fallback
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        return "I am sorry! I don't understand your question."
    else:
        return sent_tokens[idx]

# Flask route for chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    if user_input:
        response_text = response(user_input)
        try:
            sent_tokens.remove(user_input.lower())  # Remove user input from sent_tokens
        except ValueError:
            pass  # Handle case where user_response is not found in sent_tokens
        return jsonify({"response": response_text})
    return jsonify({"response": "Please provide a message to get a response."})

if __name__ == '__main__':
    app.run(debug=True)
