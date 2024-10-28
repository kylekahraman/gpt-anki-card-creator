import csv

with open("anki_flashcards.txt", "r") as file:
    lines = file.readlines()

flashcards = []
current_question = None

for line in lines:
    line = line.strip()
    if line.startswith("Q:"):
        current_question = line[3:].strip()
    elif line.startswith("A:") and current_question:
        answer = line[3:].strip()
        flashcards.append((current_question, answer))
        current_question = None

with open("anki_flashcards.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(flashcards)

print("Flashcards successfully saved as anki_flashcards.csv")
