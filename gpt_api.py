from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken
from typing import List, Optional, Tuple

load_dotenv()
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)

system_prompt = """
    You are a helpful assistant with three tasks. Follow only the task explicitly provided by the user in each step:

    1. **Task 1**: Summarize lecture transcripts by creating a structured and detailed overview organized by thematic sections.

    2. **Task 2**: Based on the summary, generate question-answer pairs suitable for Anki flashcards. This task should be done only when explicitly requested.

    3. **Task 3**: Based on the lecture transcript, the summary and the anki flashcards, generate a podcast script designed for students.

    # Important Instructions
    - Do not attempt more than one tasks at once. 
    - Wait for a specific user request before starting each task, and only proceed with the designated task.
    - Ensure clarity, accuracy, and completeness in all tasks, following the structure and guidelines outlined below.

    --- 

    # Steps for each task

    ## Task 1: Summary Creation
    Summarize lecture content into a clear, concise overview organized by chronologic thematic sections, preserving the original flow, and detailing essential information.

    ### Steps for Task 1: Summary Creation
    1. **Read and Understand**: Thoroughly read the provided lecture transcript to grasp its main topics, themes, and structure.
    2. **Identify Themes**: Identify and categorize key themes and sections in the transcript.
    3. **Create a table of contents**: List the main topics and themes covered in the lecture.
    4. **Flag Gaps or Errors**: Detect gaps or unclear parts in the transcript where content is missing, or words do not fit the context.
        - Example: "The text mentions a '_____' which seems out of place."
        - Example: "These are, These can be found underneath the salamence, which is to be part of the salamence. And the medial GI here in the lateral geniculate body, in the word of retinal cells, are basically synapses on two. So it's more of a visual processing." Salamence should be Thalamus, as the text is about brain structures.
    5. **Prioritize Important Information**: Based on terminology and content, summarize each section by keeping essential details and avoiding overly short summaries that omit critical information.
        - The lecturer might use words to indicate which important is important and which not. For example: "crucial", "important", "key", "essential"
        - The lecturer might talk about a topic for a long time, which indicates that the topic is important.
        - The lecturer returns to a topic, which indicates that the topic is important.
        - The lecturer highlights relationships between concepts, which indicates that the relationship and the topics involved are important.
    6. **Summarize**: Create a structured summary organized around the identified themes. Each thematic section should include:
        - An introductory sentence summarizing the main topic.
        - Key concepts, essential details, and terminology.
        - Relevant examples or applications, and relationships between concepts if discussed.
        - The summary should mirror the logical flow of the lecture and include only critical information, without extrapolations or interpretations.
        - List any transcription errors or gaps detected.
        - Ignore anything that denotes who is speaking in the transcript. This is from the transcription software and thus not important for the summary. However do destinguish if important bits are said by one person or not, to identify the lecturer and ignore other people speaking that the transcription software picked up. 
    
    ## Task 2: Flashcard Generation
    Based on a summary, generate Anki-style question-answer pairs to assist in learning and recall. Each question-answer pair should focus on a single key concept, definition, or relationship, ensuring clarity and relevance.

    ### Steps for Task 2: Flashcard Generation
    1. **Generate Questions and Answers**: Based on the provided summary, create Anki-style question-answer pairs, each focusing on a specific concept, definition, or relationship from the content.
        - **Questions**: Formulate questions that are clear, concise, and challenge the learner's understanding. Questions should be concise and target one main idea or detail.
        - **Answers**: Provide detailed answers that accurately represent the information from the summary.
    2. **Structure**: Format each question and answer pair as follows for clarity and ease of import into Anki:
        - "Q: [question]"
        - "A: [answer]"
    
    ## Task 3: Podcast Script Generation
    Mix the lecture transcript, its summary, and the Anki-Style Flashcards into an engaging, listener-friendly podcast script designed for students. The goal is to provide a complete script for an episode that follows the lecture's flow, edcuates listeners, and maintains their interest.

    ### Steps for Task 3: Podcast Script Generation
    1. **Structure and Flow**: 
        - Organize the script into thematic sections that align with the lecture's content.
        - Maintain a logical flow that follows the lecture's progression, ensuring each section has a clear purpose and builds on the previous one.
    2. **Incorporate Anki Q&A**:
        - Use Anki-style questions within each section conversationally, as if prompting listeners to think.
        - Ensure the script provides answers or hints so listeners can follow and engage actively.
    3. **Simplify and Engage**:
        - Use accessible language, breaking down complex terms for easier understanding
        - Keep the tone conversational and engaging, suitable for a podcast lecture format.
        - Add brief verbal transitions and reflective pauses to help listeners retain information.
    4. **Level of Detail**:
        - Maintain detail that serves the learning objective. Avoid overly complex explanations but ensure that each point is covered in enough depth to educate the listener effectively.
        - If points are detailed, consider breaking them down into smaller, digestible segments.
        - Define terms or concepts if they are crucial to understanding the topic or appear for the first time.
    ---

    # Output Format

    - **Summary**: 
    Structure in thematic sections, each consisting of a paragraph or bullet points.
    - **Question-Answer Pairs**: 
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
    - Follow the users' instructions for each task and avoid combining tasks.
    - Ensure adhering to the structure and organization for better understanding and retention.
    - Focus on clarity and relevance of both summary and questions to assist in learning and recall.
    - The output for the flashcards should only be the question-answer pairs separated by a comma and no additional text.
    """

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_chat_completion(messages, model="gpt-4o-mini", temperature=0.3, max_tokens=16384):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

def tokenize(text: str):
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return encoding.encode(text)

def chunk_on_delimiter(input_string: str, max_tokens: int, delimiter: str) -> List[str]:
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True
    )
    if dropped_chunk_count > 0:
        print(f"warning: {dropped_chunk_count} chunks were dropped due to overflow")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks

def combine_chunks_with_no_minimum(
    chunks: List[str],
    max_tokens: int,
    chunk_delimiter="\n\n",
    header: Optional[str] = None,
    add_ellipsis_for_overflow=False,
) -> Tuple[List[str], List[int]]:
    dropped_chunk_count = 0
    output = []  # list to hold the final combined chunks
    output_indices = []  # list to hold the indices of the final combined chunks
    candidate = (
        [] if header is None else [header]
    )  # list to hold the current combined chunk candidate
    candidate_indices = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]
        if len(tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
            print(f"warning: chunk overflow")
            if (
                add_ellipsis_for_overflow
                and len(tokenize(chunk_delimiter.join(candidate + ["..."])))
                <= max_tokens
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue  # this case would break downstream assumptions
        # estimate token count with the current chunk added
        extended_candidate_token_count = len(
            tokenize(chunk_delimiter.join(candidate + [chunk]))
        )
        # If the token count exceeds max_tokens, add the current candidate to output and start a new candidate
        if extended_candidate_token_count > max_tokens:
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
            candidate = chunk_with_header  # re-initialize candidate
            candidate_indices = [chunk_i]
        # otherwise keep extending the candidate
        else:
            candidate.append(chunk)
            candidate_indices.append(chunk_i)
    # add the remaining candidate to output if it's not empty
    if (header is not None and len(candidate) > 1) or (
        header is None and len(candidate) > 0
    ):
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)
    return output, output_indices, dropped_chunk_count


def chunk_text(text: str, max_tokens: int, delimiter: str = "."):
    words = text.split(delimiter)
    chunks, current_chunk = [], []
    current_token_count = 0
    for word in words:
        current_token_count += len(tokenize(word + delimiter))
        if current_token_count > max_tokens:
            chunks.append(delimiter.join(current_chunk))
            current_chunk = [word]
            current_token_count = len(tokenize(word + delimiter))
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(delimiter.join(current_chunk))
    return chunks

def summarize(
    text: str,
    detail: float = 0.5,
    model: str = "gpt-4o-mini",
    additional_instructions=None,
    recursive_summary: bool = True,
    minimum_chunk_size: Optional[int] = 500,
    chunk_delimiter: str = ".",
    verbose=False,
):
    assert 0 <= detail <= 1, "Detail level must be between 0 and 1."

    max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    document_length = len(tokenize(text))
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        print(f"Chunk lengths are {[len(tokenize(x)) for x in text_chunks]}")

    
    system_message_content = """
    You are an assistant specializing in summarizing long lecture transcripts. Each summary is part of one continuous lecture, which means that all chunks together represent a single lecture transcript, not multiple lectures.

    # Summary Creation Task
    Summarize lecture content into a clear, concise overview organized by chronologic flow of the lecture, detailing essential information without repeating or starting new lecture sections for each chunk.

    # Steps for Summary Creation Task
    1. **Read and Understand**: Thoroughly read each part to grasp its main topics, themes, and structure.
    2. **Identify Themes and Organize by Continuity**: Identify key themes and categorize sections logically within the overall context. Ensure each section flows naturally into the next.
    3. **Flag Gaps or Errors**: Identify and note any gaps, unclear parts, or unusual words in the transcript. Treat these observations as continuity points that need attention rather than new lecture breaks.
        - Example: "The text mentions a '_____' which seems out of place."
        - Example: "These are, These can be found underneath the salamence, which is to be part of the salamence. And the medial GI here in the lateral geniculate body, in the word of retinal cells, are basically synapses on two. So it's more of a visual processing." Salamence should be Thalamus, as the text is about brain structures.
        - Feel free to think of what would rather fit in the context of the lecture and replace that, while still writing it in the section on Gaps and Errors.
    4. **Prioritize Important Information**: Based on terminology and content of the lecturer, summarize each section by keeping essential details and avoiding overly short summaries that omit critical information. avoid repeating introductory information as each chunk is part of one continuous lecture.
        - The lecturer might use words to indicate which important is important and which not. For example: "crucial", "important", "key", "essential"
        - The lecturer might talk about a topic for a long time, which indicates that the topic is important.
        - The lecturer returns to a topic, which indicates that the topic is important.
        - The lecturer highlights relationships between concepts, which indicates that the relationship and the topics involved are important.
        - If something is important, put a "(important!)" next to it.
        - Don't say things like "the lecturer says..." or "the speaker says..." instead, just present the information as if someone took notes on the content of the lecture.
    5. **Summarize**: Create a structured summary organized around the identified themes. Each thematic section should include:
        - An introductory sentence summarizing the main topic.
        - Key concepts, essential details, and terminology.
        - Relevant examples or applications, and relationships between concepts if discussed.
        - The summary should mirror the logical flow of the lecture and include only critical information, without extrapolations or interpretations.
        - List any transcription errors or gaps detected, noting only errors that affect the logical flow.
        - Ignore details that identify individual speakers unless essential to understanding the lecture content.

    # IMPORTANT
    - You always only get one lecture. Not more. Always create one table of contents for the whole lecture.
    - If you think there are multiple lectures, ignore it and summarize the whole text as one lecture.
    - Stop saying things like "the lecture progresses" or "the discussion shifts" and just summarize the content.

    # Examples
    **Summary Example:**
    - **Introduction to Photosynthesis**
        - Photosynthesis is the process by which plants convert light energy into chemical energy.
    - **Key Stages**
        - Light-dependent reactions: occur in the thylakoid membranes and produce ATP and NADPH.
        - Light-independent reactions: also known as the Calvin Cycle, occur in the stroma and produce glucose.
    
    """
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for chunk in text_chunks:
        if recursive_summary and accumulated_summaries:
            accumulated_summaries_string = "\n\n".join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
        else:
            user_message_content = chunk
        
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content},
        ]
        response = get_chat_completion(messages, model=model, temperature=0.3)
        accumulated_summaries.append(response)

    final_summary = "\n\n".join(accumulated_summaries)
    return final_summary


def summarize_text(text, system_prompt):

    summary_prompt = """
    Focus on task 1 only. You will receive a summary from a lecture transcription and clean that up into a more coherent format. Do not create any flashcards in this step.

    # Instructions
    1. **Condense the information** by including only what is already present in the lecture transcript, without adding new content.
    2. **Organize the summary into clear thematic sections.** Each section should begin with an overview of the main topic, followed by key concepts, essential details, terminology, and relevant examples.
    3. **Reorganize sections** Some sections are repeated or in a chaotic format. Reorganize them to be coherent and together. 
    3. **Highlight relationships between concepts when present, and eliminate redundant information unless it emphasizes critical points.** Ensure the summary mirrors the logical flow of the original lecture, capturing all main points accurately and completely.
    4. **Maintain the original flow and structure of the lecture**, ensuring that all main points are accurately and completely represented, without any extrapolation or omission of essential content.

    Output Format:
    - **Introduction to Photosynthesis**
        - Photosynthesis is the process by which plants convert light energy into chemical energy.
    - **Key Stages**
        - Light-dependent reactions: occur in the thylakoid membranes and produce ATP and NADPH.
        - Light-independent reactions: also known as the Calvin Cycle, occur in the stroma and produce glucose.

    Lecture summary:
    {text}""".format(text=text)
    
    completion_summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary_prompt},
        ],
        temperature=0.2,
        max_tokens=16384,
    )

    summary = completion_summary.choices[0].message.content
    return summary

def generate_flashcards(summary, system_prompt):
    flashcard_prompt = """
    Focus on Task 2 only: Based on the provided summary, generate question-answer pairs suitable for Anki flashcards. Do not summarize further or add any new sections. Mix it up between creating regular Q&A pairs and Close-Deletion-Cards.

    # Instructions
    1. **Create question-answer pairs** that each focus on a single key concept, definition, or relationship highlighted in the summary.
    2. **Formulate each question concisely to cover important concepts or details, avoiding overly broad or vague phrasing.** For example, rather than "What is the nervous system?", specify "What are the main functions of the nervous system?" 
    3. **Provide direct, clear answers that reinforce understanding of each question's topic. Ensure each answer is complete and concise.**
    4. **Structure the output in a clear Question-Answer format, suitable for direct import into Anki, with each question and answer on a new line.**
        - Output format: "Q: [question]" and "A: [answer]" for each pair and close-deletion cards.

    # Important
    - Ensure that the output only contains the question-answer pairs and no additional text separated by a comma.
    - Create unique pairs only.
    - **Do Not Duplicate Prompts or Responses**: Ensure that the Q&A structure is clear and concise, with each question and answer formatted exactly as follows:
        - "Q: [question]"
        - "A: [answer]"
    **Avoid Repetitions in Wording**: Do not repeat or duplicate any question-answer phrases.
    **Stick to Essential Information**: Only include relevant details directly related to the question to avoid overcomplicating answers.
    
    # Expected Output
    - Q: Where are the cell bodies of sensory neurons located, and what structural classification do they have?
    - A: Sensory neurons have their cell bodies in the dorsal root ganglia and are classified as pseudo-unipolar neurons, with a single neurite that bifurcates.

    - Q: What is the significance of the anterior and posterior neuropores closing during development?
    - A: The closure of the anterior and posterior neuropores is essential to prevent birth defects.

    # Unexpted Output
    - Q: Q: What is the significance of the anterior and posterior neuropores closing during development?
    - A: The closure of the anterior and posterior neuropores is essential to prevent birth defects.

    Summary:
    {summary}
    """.format(summary=summary)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": flashcard_prompt},
    ]

    flashcards = get_chat_completion(messages=messages, temperature=0.1)

    # flashcards = response.choices[0].message.content
    return flashcards

def generate_regular_podcast_script(podcast_prompt, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": podcast_prompt}
    ]
    podcast_script = get_chat_completion(
        messages=messages, temperature=0.3)
    return podcast_script

def generate_podcast_script(
    text: str,
    detail: float = 0.5,
    model: str = "gpt-4o-mini",
    additional_instructions=None,
    recursive_summary: bool = True,
    minimum_chunk_size: Optional[int] = 500,
    chunk_delimiter: str = ".",
    verbose=False,
):
    assert 0 <= detail <= 1, "Detail level must be between 0 and 1."

    max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    document_length = len(tokenize(text))
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        print(f"Chunk lengths are {[len(tokenize(x)) for x in text_chunks]}")

    system_message_content = """
    You are a helpful assistant that creates a podcast script based on a lecture transcript, its summary, and its Anki-style flashcards. The goal is to make this script suitable for students who will listen to it as a complete and engaging episode. Follow the thematic structure of the transcript and provide a coherent flow that maintains interest and facilitates learning. Here are the details:

    # Instructions
    1. **Use the Original Transcript** to maintain the lecture's natural progression and flow.
    2. **Organize by Thematic Sections**: 
        - Use the summary to create concise explanations of each thematic section.
        - Structure the content in a way that students can follow and retain information easily.
    3. **Integrate Anki Q&A Pairs**: 
        - Use the Anki questions in a conversational manner to prompt listeners to think about the concepts.
        - Ensure the script provides enough context and answers or hints for each question.
    4. **Engagement and Clarity**:
        - Write in a conversational and engaging tone suitable for an audio format.
        - Add short transitions and pauses for reflection, so listeners have time to process important points.
    5. **Level of Detail**:
        - Maintain a balance between depth and simplicity, ensuring the script is informative but accessible to students.
    
    ## Podcast Script Generation Task
    Mix the lecture transcript, its summary, and the Anki-Style Flashcards into an engaging, listener-friendly podcast script designed for students. The goal is to provide a complete script for an episode that follows the lecture's flow, edcuates listeners, and maintains their interest.

    ### Steps for Podcast Script Generation Task
    1. **Structure and Flow**: 
        - Organize the script into thematic sections that align with the lecture's content.
        - Maintain a logical flow that follows the lecture's progression, ensuring each section has a clear purpose and builds on the previous one.
        - Use the summary to guide the script's structure and ensure all key points are covered.
        - First outline the main topics and themes covered in the lecture.
    2. **Incorporate Anki Q&A**:
        - Use Anki-style questions within each section conversationally, as if prompting listeners to think.
        - Ensure the script provides answers or hints so listeners can follow and engage actively.
    3. **Simplify and Engage**:
        - Use accessible language, breaking down complex terms for easier understanding
        - Keep the tone conversational and engaging, suitable for a podcast lecture format.
        - Add brief verbal transitions and reflective pauses to help listeners retain information.
    4. **Level of Detail**:
        - Maintain detail that serves the learning objective. Avoid overly complex explanations but ensure that each point is covered in enough depth to educate the listener effectively.
        - If points are detailed, consider breaking them down into smaller, digestible segments.
        - Define terms or concepts if they are crucial to understanding the topic or appear for the first time.
    ---
    # Important Information
    - The script is made to read out loud. Don't include any "Music", "Sound Effects", or "Background Noise" in the script.
    - Don't include anything like "Host" or "Question for the Audience"
    - The output is just the podcast script for me to read out loud. Don't make up an imaginary podcast with a name or episodes or anything else.

    """
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for chunk in text_chunks:
        if recursive_summary and accumulated_summaries:
            accumulated_summaries_string = "\n\n".join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
        else:
            user_message_content = chunk

        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content},
        ]
        response = get_chat_completion(messages, model=model)
        accumulated_summaries.append(response)

    podcast_script = "\n\n".join(accumulated_summaries)
    return podcast_script