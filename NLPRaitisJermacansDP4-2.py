def varda_biezums():
    import nltk
    nltk.download('punkt_tab')
    nltk.download('punkt')

    from nltk.tokenize import word_tokenize
    from collections import Counter

    text = "Mākoņainā dienā kaķis sēdēja uz palodzes. Kaķis domāja, kāpēc debesis ir pelēkas. Kaķis gribēja redzēt sauli, bet saule slēpās aiz mākoņiem."

    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    word_frequency = Counter(words)

    print("Vārdu biežuma analīze:")
    for word, freq in word_frequency.items():
        print(f"{word}: {freq}")

def valodas_noteiksana():
    text = "Mākoņainā dienā kaķis sēdēja uz palodzes. Kaķis domāja, kāpēc debesis ir pelēkas. Kaķis gribēja redzēt sauli, bet saule slēpās aiz mākoņiem."
    from langdetect import detect_langs
    try: 
        langs = detect_langs(text) 
        for item in langs: 
             return item.lang, item.prob 
    except: return "err", 0.0
    
def salidzinat_tekstus():
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from collections import Counter

    teksts1 = "Rudens lapas ir dzeltenas un oranžas. Lapas klāj zemi un padara to krāsainu."
    teksts2 = "Krāsainas rudens lapas krīt zemē. Lapas ir oranžas un dzeltenas."  
    teksts1_vardi = [word.lower() for word in word_tokenize(teksts1) if word.isalpha()]
    teksts2_vardi = [word.lower() for word in word_tokenize(teksts2) if word.isalpha()]


    kopigie_vardi = set(teksts1_vardi).intersection(set(teksts2_vardi))

    total_unique_words = len(set(teksts1_vardi).union(set(teksts2_vardi)))
    similarity_percentage = (len(kopigie_vardi) / total_unique_words) * 100
    
    return similarity_percentage

def noskanojuma_analize():
    import spacy

    nlp = spacy.load("en_core_web_sm")

    sentiment_analyzer = nlp.create_pipe("sentiment_analyzer")

    nlp.add_pipe(sentiment_analyzer)

    text = "I love this product! It is the best thing I have ever bought."

    sentiment = sentiment_analyzer(text)

    print(sentiment.score)
def tirit_tekstu():
    import re
    import nltk
    from nltk.tokenize import word_tokenize


    nltk.download('punkt')


    nteksts = "@John: Šis ir lielisks produkts!!! Vai ne? 👏👏👏 http://example.com"

  
  
    nteksts = re.sub(r'@\w+', '', nteksts)      
    nteksts = re.sub(r'http\S+', '', nteksts)  
    nteksts = re.sub(r'[^\w\s]', '', nteksts)  
    nteksts = nteksts.lower() 
    return nteksts
def teksta_rezumesana():
    import random
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.probability import FreqDist

    text = """
Latvija ir valsts Baltijas reģionā. Tās galvaspilsēta ir Rīga, kas ir slavena ar savu vēsturisko centru un skaistajām ēkām. Latvija robežojas ar Lietuvu, Igauniju un Krieviju, kā arī tai ir piekļuve Baltijas jūrai. Tā ir viena no Eiropas Savienības dalībvalstīm.
"""
    
    
    teikumi = sent_tokenize(text)
    print('teikumi:', teikumi)
    teikumu_skaits = 3
    print('teikumu_skaits:', teikumu_skaits)
    vardi = word_tokenize(text.lower())
    print('vardi:', vardi)
    

    fdist = FreqDist(vardi)
    print('fdist:', fdist)

    teikumu_vertejums = [sum(fdist[word] for word in word_tokenize(sentence.lower()) if word in fdist)
                       for sentence in teikumi]
    
    print('teikumu_vertejums:', teikumu_vertejums)
    
    teikumu_vertejums = list(enumerate(teikumu_vertejums))
    print('teikumu_vertejums:', teikumu_vertejums)

    kartoti_teikumi = sorted(teikumu_vertejums, key=lambda x: x[1], reverse=True)
    print('kartoti_teikumi:', kartoti_teikumi)

    random_teikumi = random.sample(kartoti_teikumi, teikumu_skaits)


    issuma_teikumi = sorted(random_teikumi, key=lambda x: x[0])

    saissinajums = ' '.join([teikumi[i] for i, _ in issuma_teikumi])


    print(saissinajums)

def frazu_atpazisana():
    import spacy

    nlp = spacy.load("en_core_web_sm")

    ner_categories = ["PERSON", "ORG"]

    text = "President Egils Levits participated in an event organized by he University of Latvia"

    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        if ent.label_ in ner_categories:
            entities.append((ent.text, ent.label_))
    for entity, category in entities:
        print(f"{entity}: {category}")

def sentiments():
    import spacy
    from spacytextblob.spacytextblob import SpacyTextBlob

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    text = 'This product is amazing I am very satisfied'
    doc = nlp(text)
    print(doc._.blob.polarity)

    match doc._.blob.polarity:
        case polarity if polarity > 0.1:
            print("Positive")
        case polarity if polarity < -0.1:
            print("Negative")
        case _:
            print("Neutral")

    text = 'I am disapointed in the product it does not fit the description'
    doc = nlp(text)
    print(doc._.blob.polarity)

    match doc._.blob.polarity:
        case polarity if polarity > 0.1:
            print("Positive")
        case polarity if polarity < -0.1:
            print("Negative")
        case _:
            print("Neutral")

    text = 'Neutral product, nothing much about it'
    doc = nlp(text)
    print(doc._.blob.polarity)

    match doc._.blob.polarity:
        case polarity if polarity > 0.1:
            print("Positive")
        case polarity if polarity < -0.1:
            print("Negative")
        case _:
            print("Neutral")

def iegulsana():
    import spacy
    nlp_lg = spacy.load('en_core_web_sm')

    text = "house apartment sea"
    doc = nlp_lg(text)
    
    house = doc[0]  
    apartment = doc[1] 
    sea = doc[2]  

    print("Māja un dzīvoklis", house.similarity(apartment))
    print("Māja un jūra", house.similarity(sea))
    print("Jūra un dzīvoklis", sea.similarity(apartment))
    
