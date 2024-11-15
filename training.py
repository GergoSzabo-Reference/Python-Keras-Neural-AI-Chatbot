import json, numpy as np, spacy, random

# Sequential() -> neurális hálózat felépítése
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nlp = spacy.load("en_core_web_sm")

INTENTS_PATH = "C:\\Users\\h3d3r\\Desktop\\egyetem\\prog2\\python\\Keras Chatbot\\intents.json"
SAVE_PATH = "C:\\Users\\h3d3r\\Desktop\\egyetem\\prog2\\python\\Keras Chatbot\\training_data.npz"

intents = json.loads(open(INTENTS_PATH).read())

all_words, tags, documents = [], [], []
ignore_letters = ['?', '!', '.', ',']

for intent in intents["intents"]:
    for sentence in intent["patterns"]:
        doc = nlp(sentence)
        word_list = [token.lemma_ for token in doc if token.is_alpha]  # Lemmatizálás és csak az alfanumerikus tokenek
        all_words.extend(word_list)

        documents.append((word_list, intent["tag"]))

        if intent["tag"] not in tags:
            tags.append(intent["tag"])


print(documents)

# A szavak és a címkék halmazának rendezese (arra az esetre, ha lenne duplikátum)
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# mentés
np.savez(SAVE_PATH, all_words, tags)

# gépi tanulás előkészítése - numerizálni kell az adatokat, hogy a modell értelmezni tudja
training = []
output_empty = np.zeros(len(tags))  # minden kategóriához létrehozunk egy elemet (0)

# Minden egyes dokumentum feldolgozása a tanító adathoz
for document in documents:
    bag = []
    word_patterns = document[0]

    for word in all_words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[tags.index(document[1])] = 1
    training.append([bag, output_row])

# Adatok véletlenszerű keverése
random.shuffle(training) # nem lesz overfitting
training = np.array(training)

x_train = training[:, 0] # szavak
y_train = training[:, 1] # címkék

# Neurális háló
model = Sequential([
    # bemeneti réteg
    # 128 neuron, BoW size -> num of inputs, 
    Dense(128, input_shape = (len(x_train[0]),), activation = 'relu'),
    # overfitting: jól teljesít training data-val de nem új data-val
    # regulization technique: 50% a neuronoknak random ignorálva (elhagyva) van egy iterációban
    Dropout(0.5),
    # rejtett réteg 64 neuronnal, better in generalizing
    Dense(64, activation = 'relu'),
    Dropout(0.5),
    # kimeneti réteg, amely a valószínűségi eloszlást adja
    Dense(len(y_train[0]), activation = 'softmax')
])

LEARNING_RATE = 0.01
MOMENTUM = 0.9

# Optimalizáló konfigurálása
sgd = SGD(learning_rate = LEARNING_RATE, momentum = MOMENTUM, nesterov = True)

# Modell összeállítása, kategóriás "kereszthentrópiás veszteségfüggvénnyel" és pontosság metrikával
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# Modell mentése
hist = model.fit(np.array(x_train), np.array(y_train), epochs = 200, batch_size = 5, verbose = 1)
model.save('chatbot_model.h5', hist)