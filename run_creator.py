from typing import List, Optional, Tuple
import os
import re
import spacy
from pathlib import Path
from gpt_api import summarize_text, generate_flashcards, system_prompt, tokenize, summarize, generate_podcast_script, num_tokens_from_string, generate_regular_podcast_script
import tiktoken

# Lade das deutsche Modell von spaCy (falls Transkript auf Deutsch)
nlp = spacy.load("en_core_web_sm")

def clean_transcript(text):
    # Entferne Timecodes im Format [00:00:00] oder (00:00)
    text = re.sub(r"\[\d{2}:\d{2}:\d{2}\]|\(\d{2}:\d{2}\)", "", text)
    return text

output_path = Path("/Users/kyle/GitHub/gpt-anki-card-creator/Neuroanatomy 2")

outstring = "Neuroanatomy Day 2"

with open(
    "/Users/kyle/GitHub/gpt-anki-card-creator/Neuroanatomy 2/Neuroanatomy 2.txt", "r"
) as file:
    raw_text = file.read()

#* currently not needed?
#clean_text = clean_transcript(raw_text)

#! Create tokenizer
num_tokens = num_tokens_from_string(raw_text, "o200k_base")
if num_tokens > 128000:
    print("Text is too long for GPT-4o-mini.")
    exit()


# summary = summarize(raw_text, detail=0.5, recursive_summary=True)
# print(summary)

# print("Summary:\n", summary)
# with open(output_path / f"{outstring}_Summary.md", "w") as file:
#     file.write(summary)

# Implement summary of summary which restructures the summary into a more coherent format and removes the errors and gaps
with open(
    "/Users/kyle/GitHub/gpt-anki-card-creator/Neuroanatomy 2/Neuroanatomy Day 2_Summary.md",
    "r",
) as file:
    summary = file.read()


# * Create summary
# * old function, dont use
new_summary = summarize_text(summary, system_prompt)
with open(output_path / f"{outstring}_SumofSum.txt", "w") as file:
    file.write(new_summary)

#* Create flashcards
flashcards = generate_flashcards(summary, system_prompt)

print("Flashcards:\n", flashcards)
with open(output_path / f"{outstring}_altFlashcards.txt", "w") as file:
    file.write(flashcards)

#* Create Podcast script
podcast_prompt = """
    I need you to create a podcast transcript with the materials below:

    **Original Transcript**:
    {transcript}

    **Summary**:
    {summary}

    **Anki-Style Flashcards**:
    {anki_flashcards}

    Ensure that the podcast script follows a narrative style and is ready to be read aloud, with thematic headings and smooth transitions between sections to enhance listener engagement.

    """.format(transcript=raw_text, summary=summary, anki_flashcards=flashcards)

podcast_script = generate_regular_podcast_script(podcast_prompt, system_prompt)
print("Podcast script:\n", podcast_script)
with open(output_path / f"{outstring}_Oldfunction-Script.md", "w") as file:
    file.write(podcast_script)


# podcast_script = generate_podcast_script(podcast_prompt, detail=0.5, recursive_summary=True)

# print("Podcast script:\n", podcast_script)
# with open(output_path / f"{outstring}_Script.md", "w") as file:
#     file.write(podcast_script)