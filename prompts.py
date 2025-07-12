from ast import List
from typing import Dict


def create_chat_prompt( user_message: str, memory_context: str, recent_context: str) -> str:
    full_prompt = """You are a helpful, friendly AI assistant with memory capabilities. You can remember information from previous conversations and use it to provide more personalized and contextual responses.

    When responding:
    - Be natural and conversational
    - Be short and specific to answer do not generete long responses if not required try to answer what user saying nothing else.
    - Strictly: use past infromation only when required, do not generate long responses if not required.

    """

    if memory_context:
        full_prompt += f"The memory context is just the past summary of user information\n{memory_context}\n"

    if recent_context:
        full_prompt += f"The recent context is just the last few user messages \n{recent_context}\n"
            
    full_prompt += "Note: This the summary and recent conversation is just the information use it only when required.\n"
    full_prompt += "Now, respond to the user's message:\n"

    full_prompt += f"\nUser: {user_message}\nAssistant:"

    return full_prompt

def form_extraction_prompt(summary, recent_messages, mt_1, mt):
    """
    Forms the prompt for the LLM, directing it to extract new, salient,
    factual information exclusively from the LATEST message exchange,
    rephrase them concisely, and return <none> if no such information exists.
    """
    prompt = (
        "You are an AI assistant designed to extract **CRUCIAL, NEW, ACTIONABLE FACTS** from conversations for memory storage.\n"
        "Your goal is to identify and extract *only* facts that are essential to remember for future interactions, focusing on user-specific details, preferences, instructions, or significant updates.\n"
        "**Your primary directive is to be precise and useful, capturing important details while avoiding noise.**\n\n"
    )

    # Place the core data directly with the strictest instructions
    prompt += "### Primary Source for Facts (Analyze ONLY This Section for Extraction):\n"
    prompt += f"User: {mt_1}\nAssistant: {mt}\n\n"
    prompt += "### End Primary Source\n\n"


    # Provide context, but with clear boundaries
    prompt += "### Contextual Information (For Background ONLY - DO NOT Extract Facts From Here):\n"
    prompt += "--- General Conversation Summary (Provides historical context; DO NOT EXTRACT FACTS):\n"
    prompt += f"{summary}\n\n"
    prompt += "--- Recent Conversation History (Provides immediate context; DO NOT EXTRACT FACTS):\n"
    if not recent_messages:
        prompt += "No recent messages.\n"
    else:
        for msg in recent_messages:
            if 'content' in msg and 'role' in msg:
                prompt += f"- {msg['role'].capitalize()}: {msg['content']}\n"
            elif 'user' in msg and 'assistant' in msg:
                prompt += f"- User: {msg['user']}\n- Assistant: {msg['assistant']}\n"
            elif 'content' in msg:
                prompt += f"- {msg['content']}\n"
    prompt += "### End Contextual Information\n\n"

    # Final, balanced task instructions
    prompt += (
        "## Task: Extract CRUCIAL, NEW, ACTIONABLE Facts\n"
        "1.  Your sole focus is to analyze **EXCLUSIVELY** the 'Primary Source for Facts' section (the latest User-Assistant exchange) to identify new facts.\n"
        "2.  **DO NOT** extract any facts or information from the 'Contextual Information' sections (Summary or Recent Conversation History). These are for understanding the ongoing conversation, not for new fact extraction.\n"
        "3.  Extract only **specific, verifiable, and important details** that establish **user preferences, explicit instructions, or new factual information** directly relevant to the user or task. This includes things like:\n"
        "    - User's name, age, location, or contact info if provided.\n"
        "    - Stated preferences (e.g., 'I prefer coffee', 'I like dark mode').\n"
        "    - Direct instructions or requirements (e.g., 'Remind me tomorrow', 'I need the report by Friday').\n"
        "    - Key information relevant to a specific ongoing task.\n"
        "4.  **Concise Rephrasing:** If a fact is extracted, **rephrase it to be as short, simple, and informative as possible.** Eliminate all filler words. Focus on keywords and direct statements. Grammar is secondary to conciseness. For example, turn 'The user mentioned that their name is John' into 'User name: John'. Combine related facts into a single, concise sentence.\n"
        "5.  **When to return <none>:** If the 'Primary Source for Facts' contains *only* content that is not a crucial, new, actionable fact, you **MUST** output `<none>` and nothing else. This applies to content types such as:\n"
        "    - **Conversational Overhead:** Greetings, small talk, pleasantries, or simple acknowledgments (e.g., 'Hi', 'Hello', 'How are you?', 'Okay', 'Got it', 'Thanks', 'You're welcome').\n"
        "    - **General Information Queries:** Questions about general knowledge, current events, or topics that do not reveal specific user preferences, instructions, or personal facts (e.g., 'What is the capital of France?', 'Tell me about AI', 'What's the weather like?').\n"
        "    - **Redundant Information:** Information that is already known and does not provide a new update or clarification about the user or task.\n"
        "    - **Unactionable Statements:** Statements that don't require any memory or follow-up action.\n"
        "    Do not explain why you returned `<none>`.\n\n"
        "**Example Output Format for Extracted Facts:**\n"
        "- User's name: John.\n"
        "- Prefers: dark theme.\n"
        "- Needs: report by Friday.\n"
        "- Task: book flight to Delhi on Monday.\n"
        "\n"
        "Extracted Facts (or <none>):\n"
    )

    return prompt


def create_update_prompt(candidate_fact: str, similar_memories) -> str:
    print("similar_memories:")
    print(similar_memories)
    prompt = f"""You are an intelligent memory management system designed to process new information into a knowledge base. Your task is to analyze a 'Candidate Fact' and compare it meticulously with a list of 'Existing Similar Memories' to determine the precise operation required.

**Your Guiding Principle:** Be extremely selective. **Avoid adding redundant information.** Only add if the 'Candidate Fact' introduces a truly unique and previously unrecorded piece of information. Prioritize updating existing memories if the candidate fact refines or replaces them, even with slight wording differences.

---
## Candidate Fact to Evaluate
**Candidate Fact:** {candidate_fact}

---
## Existing Similar Memories (for comparison)
"""

    if similar_memories:
        for i, memory in enumerate(similar_memories, 1):
            prompt += f"Memory ID: {memory.get('memory_id', 'N/A')}\n" \
                      f"Content: {memory.get('content', '')}\n" \
                      f"Similarity Score: {memory.get('score', 0):.3f}\n" \
                      f"---\n" # Separator for clarity between memories
    else:
        prompt += "No similar memories found. This is highly likely a new fact. Proceed with ADD.\n---\n" # Strengthen "new fact"

    prompt += """
---
## Defined Operations

* **ADD:** Choose this operation **ONLY IF** the 'Candidate Fact' introduces a genuinely **NET-NEW CONCEPT or PIECE OF INFORMATION** that is not mentioned, implied, or contradicted by *any* of the 'Existing Similar Memories'. This should be used sparingly, as a last resort, after ruling out UPDATE, DELETE, and NOOP.
    * **Condition:** The candidate fact represents a distinct, never-before-recorded piece of actionable intelligence.
    * **Example:** Existing: "User prefers coffee." Candidate: "User wants a new phone." -> ADD.
    * **Output for ADD:** {"operation": "ADD", "target_memory_id": "", "updated_content": null}

* **UPDATE:** Choose this operation **PRIORITIZE THIS** if the 'Candidate Fact' provides **additional, refined, or slightly different details** about an existing, highly similar memory. This includes cases where the wording is different but the core meaning or subject is the same. It enhances, clarifies, or subtly modifies an existing record without outright contradicting it.
    * **Condition:** The candidate fact adds specific, complementary, or slightly rephrased information to an existing memory, making the existing memory more complete or accurate.
    * **Example 1 (Refinement):** Existing: "User lives in New York." Candidate: "User lives in New York, specifically in Brooklyn." -> UPDATE. New content: "User lives in New York, Brooklyn."
    * **Example 2 (Rewording/Completion):** Existing: "User is 30 years old." Candidate: "My age is 30." -> UPDATE. New content: "User is 30 years old." (Even if semantically same, rephrase to standard).
    * **Example 3 (Small detail added):** Existing: "User wants pizza." Candidate: "I want a large pepperoni pizza." -> UPDATE. New content: "User wants a large pepperoni pizza."
    * **Output for UPDATE:** {"operation": "UPDATE", "target_memory_id": "ID_OF_MEMORY_TO_UPDATE", "updated_content": "Revised combined and concise content"}

* **DELETE:** Choose this operation **ONLY IF** the 'Candidate Fact' *directly and unequivocally contradicts or invalidates* an *existing, highly similar memory*. The new information makes an old memory factually incorrect or completely irrelevant.
    * **Condition:** The candidate fact makes an existing memory explicitly false or obsolete.
    * **Example:** Existing: "User's name is John." Candidate: "My name is actually Mike, not John." -> DELETE "User's name is John".
    * **Output for DELETE:** {"operation": "DELETE", "target_memory_id": "ID_OF_MEMORY_TO_DELETE", "updated_content": null}

* **NOOP:** Choose this operation **ONLY IF** the 'Candidate Fact' is **EXACTLY THE SAME (or near-exact semantic duplicate)** as an existing memory, provides **NO NEW INFORMATION WHATSOEVER**, or is **insignificant/redundant** given existing memories. This means no action is required because the memory already exists in a sufficient form.
    * **Condition:** The candidate fact offers no meaningful addition or change to an existing memory. It's a true repetition or irrelevant detail.
    * **Example 1:** Existing: "User likes coffee." Candidate: "I really like coffee." -> NOOP.
    * **Example 2:** Existing: "User lives in Berlin." Candidate: "The user's residence is Berlin." -> NOOP. (Same fact, different phrasing, no new info).
    * **Output for NOOP:** {"operation": "NOOP", "target_memory_id": "", "updated_content": null}

---
## Instructions for Decision Making

1.  **Prioritize Operations (in order): NOOP > DELETE > UPDATE > ADD.** If a 'Candidate Fact' fits the criteria for NOOP, choose NOOP. If not, check DELETE. If not, check UPDATE. Only if none of the above apply, choose ADD.
2.  **Strict Comparison:** Carefully compare the 'Candidate Fact' with *each* 'Existing Similar Memory'. Consider semantic meaning, not just exact wording.
3.  **Combine and Condense for UPDATE:** For **UPDATE**, you *must* generate a new, `updated_content` string that thoughtfully combines the information from the `Candidate Fact` and the `target_memory`. This `updated_content` should be **as concise and informative as possible**, removing redundancy, similar to the fact extraction process. (e.g., combine "User lives in New York" and "User lives in Brooklyn" into "User lives in New York, Brooklyn").
4.  **Target ID for UPDATE/DELETE:** If **UPDATE** or **DELETE**, you *must* identify the `target_memory_id` of the specific memory to be acted upon.
5.  **Output Format Adherence:** Provide your response **ONLY as a JSON object**. Do not include any other text, explanations, or conversational fillers outside the JSON.

---
Extracted JSON operation:
"""

    return prompt


def create_summary_prompt(memories_list):
    # This initial instruction is good, sets the role
    memories_section = "You are a summary writer focused on extreme brevity and key facts.\n"
    memories_section += "Extract only critical, actionable information from the following memories:\n"
    memories_section += "\n--- Memories Start ---\n"
    for memory in memories_list:
        memories_section += f"- {memory['content'].strip()}\n" # Added hyphen for bullet, strip for clean lines
    memories_section += "--- Memories End ---\n\n"

    prompt = (
        f"{memories_section}"
        "Rules for Summary:\n"
        "1. **Crucial Facts Only:** Include only facts that absolutely *must* be remembered for future interactions (e.g., user preferences, explicit instructions, names, important decisions).\n"
        "2. **Minimalist:** Eliminate all filler words, greetings, small talk, and conversational flow. Get straight to the point.\n"
        "3. **Short Phrases/Keywords:** Use short phrases, keywords, or bullet points (if natural) instead of full sentences. Grammar is secondary to conciseness.\n"
        "4. **No Introduction/Conclusion:** Start directly with the facts. Do not write 'Here is a summary' or similar.\n"
        "5. **Paragraph Format:** Despite brevity, output as a single, dense paragraph.\n"
        "6. **Example of desired style: 'User name: John. Prefers dark mode. Needs report by Friday.'**\n\n"
        "Generate the extremely concise summary based on these rules. If no crucial facts exist, output 'No key info yet.'\n"
        "Summary:"
    )
    return prompt