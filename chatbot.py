import random, json, numpy as np, spacy

from tensorflow.keras.models import load_model

# Adatok betöltése
nlp = spacy.load("en_core_web_sm")

INTENTS_PATH = "C:\\Users\\h3d3r\\Desktop\\egyetem\\prog2\\python\\Keras Chatbot\\intents.json"
SAVE_PATH = "C:\\Users\\h3d3r\\Desktop\\egyetem\\prog2\\python\\Keras Chatbot\\training_data.npz"

intents = json.loads(open(INTENTS_PATH).read())

all_words, tags = np.load(SAVE_PATH)

model = load_model('chatbot_model.h5')

# lemmatizálás
def preprocess_sentence(sentence):
    doc = nlp(sentence)
    sentence_words = [token.lemma_ for token in doc if token.is_alpha]

    return sentence_words

# mondat értelmezése
def bag_of_words(sentence):
    # szavakra bontás
    sentence_words = preprocess_sentence(sentence)

    # tanító szavak hosszával feltöltjük a bag vektort
    bag = np.zeros(len(all_words), dtype=np.float32)

    for trained_word in sentence_words:
        for i, word in enumerate(all_words):
            if word == trained_word:
                bag[i] = 1

    return bag

# előrejelzi a legvalószínűbb címkét
def predict_tag(sentence):
    bow = bag_of_words(sentence) # mely szavak vannak benne a mondatban
    result = model.predict(np.array([bow]))[0] # neurális hálózaton való kiértékelés, valségi eloszlást ad vissza

    ERROR_THRESHOLD = 0.25 # mely eredmények számítanak MAGAS valségnek, csak azokat vesszük figyelembe
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD] # <0.25 szűrése, végig iterálunk a valségeken

    results.sort(key=lambda x: x[1], reverse = True) # legvalószínűbb szándékok előre helyezése
    # csökkenő sorrend
    # x[1]: aktuális valség értéke

    return_list = []

    for r in results: # legjobb eredményeket (szándékokat, azok valségeit hozzáadja egy listához)
        return_list.append({'intent': tags[r[0]], 'probability': str(r[1])}) # r[0] index, r[1] valség

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent'] # 1. intent, mert az a legvalószínűbb
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag: # megkeresi a megfelelő intentet jsonban
            result = random.choice(i['responses']) # a választ hozzá
            break

    return result

print("Chatbot: Say something! ('exit' to quit the program)")
while True:
    message = input("You: ")
    if message.lower() == "exit":
        break
    intent_list = predict_tag(message)
    response = get_response(intent_list, intents)
    print(f"Chatbot {response}")