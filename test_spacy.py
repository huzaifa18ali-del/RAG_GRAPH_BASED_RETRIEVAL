import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I want to learn AI. Math scares me. But I will try anyway.")

for sent in doc.sents:
    print(sent.text)