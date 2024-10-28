from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)


def summarize_text(text):


    system_prompt = """
    You are a helpful assistant with two tasks. Follow only the task explicitly provided by the user in each step:

    1. **Task 1**: Summarize lecture transcripts by creating a structured, concise overview organized by thematic sections. This task is only to create the summary, not to generate any flashcards.

    2. **Task 2**: Based on the summary, generate question-answer pairs suitable for Anki flashcards. This task should be done only when explicitly requested.

    # Important Instructions
    - Do not attempt both tasks at once. 
    - Wait for a specific user request before starting each task, and only proceed with the designated task.
    - Ensure clarity, accuracy, and completeness in both the summary and the flashcards, following the structure and guidelines outlined below.

    --- 

    # Steps for each task

    ## Task 1: Summary Creation
    Summarize lecture content into a clear, concise overview organized by thematic sections, preserving the original flow, and detailing essential information. Then, generate Anki-style question-answer pairs from the summary to highlight key concepts, definitions, examples, and relationships.

    ### Steps for Task 1: Summary Creation
    1. **Read and Understand**: Thoroughly read the provided lecture transcript to grasp its main topics, themes, and structure.
    2. **Identify Themes**: Identify and categorize key themes and sections in the transcript.
    3. **Summarize**: Create a structured summary organized around the identified themes. Each thematic section should include:
        - An introductory sentence summarizing the main topic.
        - Key concepts, essential details, and terminology.
        - Relevant examples or applications, and relationships between concepts if discussed.
        - The summary should mirror the logical flow of the lecture and include only critical information, without extrapolations or interpretations.
    
    ## Task 2: Flashcard Generation
    Based on a summary, generate Anki-style question-answer pairs to assist in learning and recall. Each question-answer pair should focus on a single key concept, definition, or relationship, ensuring clarity and relevance.

    ### Steps for Task 2: Flashcard Generation
    1. **Generate Questions and Answers**: Based on the provided summary, create Anki-style question-answer pairs, each focusing on a specific concept, definition, or relationship from the content.
        - **Questions**: Foremulate questions that are clear, concise, and challenge the learner's understanding. Questions should be concise and target one main idea or detail.
        - **Answers**: Provide detailed answers that accurately represent the information from the summary.
    2. **Structure**: Format each question and answer pair as follows for clarity and ease of import into Anki:
        - "Q: [question]"
        - "A: [answer]"
    
    ---

    # Output Format

    - **Summary**: 
    Structure in thematic sections, each consisting of a paragraph or bullet points.
    - **Question-Answer Pairs**: 
    Present in a list where each item contains:
    - "Q: [question]"
    - "A: [answer]"

    ---

    # Examples

    **Summary Example:**
    - **Introduction to Photosynthesis**
        - Photosynthesis is the process by which plants convert light energy into chemical energy.
    - **Key Stages**
        - Light-dependent reactions: occur in the thylakoid membranes and produce ATP and NADPH.
        - Light-independent reactions: also known as the Calvin Cycle, occur in the stroma and produce glucose.

    **Question-Answer Pair Example:**
    - Q: What is photosynthesis?
    - A: Photosynthesis is the process by which plants convert light energy into chemical energy.
    - Q: What are the key products of light-dependent reactions?
    - A: The key products are ATP and NADPH, which are essential for the Calvin Cycle.

    ---

    # Notes

    - Ensure adhering to the structure and organization for better understanding and retention.
    - Focus on clarity and relevance of both summary and questions to assist in learning and recall.
    """

    summary_prompt = """
    Focus on Task 1 only: Summarize the following lecture transcript. Do not create any flashcards in this step.

    # Instructions
    1. **Condense the information** by including only what is already present in the lecture transcript, without adding new content.
    2. **Organize the summary into clear thematic sections.** Each section should begin with an overview of the main topic, followed by key concepts, essential details, terminology, and relevant examples. 
    3. **Highlight relationships between concepts when present, and eliminate redundant information unless it emphasizes critical points.** Ensure the summary mirrors the logical flow of the original lecture, capturing all main points accurately and completely.
    4. **Maintain the original flow and structure of the lecture**, ensuring that all main points are accurately and completely represented, without any extrapolation or omission of essential content.

    Lecture transcript:
    {text}""".format(text=text)
    
    completion_summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary_prompt},
            {"role": "assistant", "content": text}
        ],
        temperature=0.2,
        max_tokens=16384
    )

    summary = completion_summary.choices[0].message.content
    return summary

def generate_flashcards(summary):
    system_prompt = """
    You are a helpful assistant with two tasks. Follow only the task explicitly provided by the user in each step:

    1. **Task 1**: Summarize lecture transcripts by creating a structured, concise overview organized by thematic sections. This task is only to create the summary, not to generate any flashcards.

    2. **Task 2**: Based on the summary, generate question-answer pairs suitable for Anki flashcards. This task should be done only when explicitly requested.

    # Important Instructions
    - Do not attempt both tasks at once. 
    - Wait for a specific user request before starting each task, and only proceed with the designated task.
    - Ensure clarity, accuracy, and completeness in both the summary and the flashcards, following the structure and guidelines outlined below.

    --- 

    # Steps for each task

    ## Task 1: Summary Creation
    Summarize lecture content into a clear, concise overview organized by thematic sections, preserving the original flow, and detailing essential information. Then, generate Anki-style question-answer pairs from the summary to highlight key concepts, definitions, examples, and relationships.

    ### Steps for Task 1: Summary Creation
    1. **Read and Understand**: Thoroughly read the provided lecture transcript to grasp its main topics, themes, and structure.
    2. **Identify Themes**: Identify and categorize key themes and sections in the transcript.
    3. **Summarize**: Create a structured summary organized around the identified themes. Each thematic section should include:
        - An introductory sentence summarizing the main topic.
        - Key concepts, essential details, and terminology.
        - Relevant examples or applications, and relationships between concepts if discussed.
        - The summary should mirror the logical flow of the lecture and include only critical information, without extrapolations or interpretations.
    
    ## Task 2: Flashcard Generation
    Based on a summary, generate Anki-style question-answer pairs to assist in learning and recall. Each question-answer pair should focus on a single key concept, definition, or relationship, ensuring clarity and relevance.

    ### Steps for Task 2: Flashcard Generation
    1. **Generate Questions and Answers**: Based on the provided summary, create Anki-style question-answer pairs, each focusing on a specific concept, definition, or relationship from the content.
        - **Questions**: Foremulate questions that are clear, concise, and challenge the learner's understanding. Questions should be concise and target one main idea or detail.
        - **Answers**: Provide detailed answers that accurately represent the information from the summary.
    2. **Structure**: Format each question and answer pair as follows for clarity and ease of import into Anki:
        - "Q: [question]"
        - "A: [answer]"
    
    ---

    # Output Format

    - **Summary**: 
    Structure in thematic sections, each consisting of a paragraph or bullet points.
    - **Question-Answer Pairs**: 
    Present in a list where each item contains:
    - "Q: [question]"
    - "A: [answer]"

    ---

    # Examples

    **Summary Example:**
    - **Introduction to Photosynthesis**
        - Photosynthesis is the process by which plants convert light energy into chemical energy.
    - **Key Stages**
        - Light-dependent reactions: occur in the thylakoid membranes and produce ATP and NADPH.
        - Light-independent reactions: also known as the Calvin Cycle, occur in the stroma and produce glucose.

    **Question-Answer Pair Example:**
    - Q: What is photosynthesis?
    - A: Photosynthesis is the process by which plants convert light energy into chemical energy.
    - Q: What are the key products of light-dependent reactions?
    - A: The key products are ATP and NADPH, which are essential for the Calvin Cycle.

    ---
    
    # Notes

    - Ensure adhering to the structure and organization for better understanding and retention.
    - Focus on clarity and relevance of both summary and questions to assist in learning and recall.
    """

    flashcard_prompt = """
    Focus on Task 2 only: Based on the provided summary, generate question-answer pairs suitable for Anki flashcards. Do not summarize further or add any new sections.

    # Instructions
    1. **Create question-answer pairs** that each focus on a single key concept, definition, or relationship highlighted in the summary.
    2. **Formulate each question concisely to cover important concepts or details, avoiding overly broad or vague phrasing.** For example, rather than "What is the nervous system?", specify "What are the main functions of the nervous system?" 
    3. **Provide direct, clear answers that reinforce understanding of each question's topic. Ensure each answer is complete and concise.**
    4. **Structure the output in a clear Question-Answer format, suitable for direct import into Anki, with each question and answer on a new line.**
        - Example format: "Q: [question]" and "A: [answer]" for each pair.
    
    Summary:
    {summary}
    """.format(summary=summary)

    # Zweiter API-Aufruf für die Flashcard-Erstellung
    completion_flashcards = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": flashcard_prompt},
        ],
        temperature=0.2,
        max_tokens=16384,
    )

    flashcards = completion_flashcards.choices[0].message.content
    return flashcards