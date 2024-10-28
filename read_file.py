import re
import spacy
from gpt_api import summarize_text, generate_flashcards


# Lade das deutsche Modell von spaCy (falls Transkript auf Deutsch)
nlp = spacy.load("en_core_web_sm")

def clean_transcript(text):
    # Entferne Timecodes im Format [00:00:00] oder (00:00)
    text = re.sub(r"\[\d{2}:\d{2}:\d{2}\]|\(\d{2}:\d{2}\)", "", text)
    return text


# def segment_text(text):
#     doc = nlp(text)
#     # Unterteile in Absätze, basierend auf Sätzen
#     sentences = [sent.text for sent in doc.sents]
#     return sentences

with open("/Users/kyle/GitHub/gpt-anki-card-creator/transcript1.txt", "r") as file:
    raw_text = file.read()

clean_text = clean_transcript(raw_text)
#segments = segment_text(clean_text)

#* Create summary
summary = summarize_text(clean_text)

print("Summary:\n", summary)
with open("summary.txt", "w") as file:
    file.write(summary)

#* Create flashcards
flashcards = generate_flashcards(summary)

print("Flashcards:\n", flashcards)
with open("anki_flashcards.txt", "w") as file:
    file.write(flashcards)