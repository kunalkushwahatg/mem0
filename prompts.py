from ast import List
from typing import Dict


def create_chat_prompt(self, user_message: str, memory_context: str, recent_context: str) -> str:
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


def form_extraction_prompt(self, summary, recent_messages, mt_1, mt):
        """
        Forms the prompt for the LLM, clearly directing it to extract facts
        only from the recent messages and the new message pair.
        """
        prompt = (
            "You are an AI assistant specializing in extracting key facts from conversations. "
            "Your primary goal is to identify the most important and salient facts "
            "**EXCLUSIVELY from  latest message exchange.**\n\n"
        )

        # Start with the most relevant data for the task
        prompt += "## Latest Conversation Exchange\n"
        prompt += f"User: {mt_1}\nAssistant: {mt}\n\n"
        prompt += "Latest conversation exchange END \n"
        prompt += "## Recent Conversation History (DO NOT EXTRACT FACTS FROM HERE - this is for overall context)\n"
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
        prompt += "\n"
        prompt += "Recent conversation history END \n"
        # Now, clearly separate the summary as *background only*.
        prompt += (
            f"--- General Conversation Summary (DO NOT EXTRACT FACTS FROM HERE - this is for overall context):\n"
            f"{summary}\n\n"
        )
        prompt += "General Conversation Summary END \n"

        # Provide very specific and strong instructions for the task
        prompt += (
            "## Task: Extract Key Facts\n"
            "From the **'Latest Conversation Exchange'** and **'Recent Conversation History'** sections ONLY with the help of summary, "
            "identify the most important and distinct facts or memories and generate only that much is required.\n"
            "Note: Summary and recent conversation history is source of help not for fact extarction so avoid extarcting fact from it \n"
            "Strictly : only generate facts not other text also focous moslty on generating new text form latest conversation exchange  \n"
            "NOTE: try to put same type of information in one sentence.\n"
            "Strictly and MUST: If there is no fact or just random conversation in the latest exchange, DO NOT generate any facts return <none>.\n"
            "Extracted Facts:\n"
        )
        
        return prompt

def create_update_prompt(self, candidate_fact: str, similar_memories: List[Dict]) -> str:
        print("similar_memories:")
        print(similar_memories)
        prompt = f"""You are an intelligent memory management system designed to process new information into a knowledge base. Your task is to analyze a 'Candidate Fact' and compare it meticulously with a list of 'Existing Similar Memories' to determine the precise operation required.

**Your Guiding Principle:** Prioritize adding genuinely new, distinct information. Only update or delete if there's a clear, strong overlap or contradiction.

---
## Candidate Fact to Evaluate
**Candidate Fact:** {candidate_fact}

---
## Existing Similar Memories (for comparison)
"""
        
        if similar_memories:
            for i, memory in enumerate(similar_memories, 1):
                # Ensure a clear format for each memory
                prompt += f"Memory ID: {memory.get('memory_id', 'N/A')}\n" \
                          f"Content: {memory.get('content', '')}\n" \
                          f"Similarity Score: {memory.get('score', 0):.3f}\n" \
                          f"---\n" # Separator for clarity between memories
        else:
            prompt += "No similar memories found. This is likely a new fact.\n---\n" # Emphasize "new fact"

        prompt += """
---
## Defined Operations

* **ADD:** Choose this operation **ONLY IF** the 'Candidate Fact' introduces genuinely *new and distinct information* that is not adequately covered, implied, or contradicted by *any* of the 'Existing Similar Memories'. This is the default if no clear update or deletion is needed.
    * **Condition:** The candidate fact brings a unique piece of information or a new perspective.
    * **Example:** "The user ordered pizza." -> No existing memory about pizza.
    * **Output for ADD:** {"operation": "ADD", "target_memory_id": "", "updated_content": null}

* **UPDATE:** Choose this operation **ONLY IF** the 'Candidate Fact' provides *additional details or refinements* to an *existing, highly similar memory*. It enhances, clarifies, or slightly modifies an existing record without contradicting it.
    * **Condition:** The candidate fact adds specific, complementary information to an existing memory.
    * **Example:** Existing: "User lives in New York." Candidate: "User lives in New York, specifically in Brooklyn." -> Update existing memory.
    * **Output for UPDATE:** {"operation": "UPDATE", "target_memory_id": "ID_OF_MEMORY_TO_UPDATE", "updated_content": "Revised combined content"}

* **DELETE:** Choose this operation **ONLY IF** the 'Candidate Fact' *directly contradicts or invalidates* an *existing, highly similar memory*. The new information makes an old memory factually incorrect or irrelevant.
    * **Condition:** The candidate fact makes an existing memory false or obsolete.
    * **Example:** Existing: "User's name is John." Candidate: "My name is actually Mike, not John." -> Delete "User's name is John".
    * **Output for DELETE:** {"operation": "DELETE", "target_memory_id": "ID_OF_MEMORY_TO_DELETE", "updated_content": null}

* **NOOP:** Choose this operation **ONLY IF** the 'Candidate Fact' is *completely redundant*, a *repetition*, or *insignificant* in the context of existing memories. No meaningful action is required.
    * **Condition:** The candidate fact provides no new information, or it's a near-exact duplicate of an existing memory.
    * **Example:** Existing: "User likes coffee." Candidate: "I really like coffee." -> No operation needed.
    * **Output for NOOP:** {"operation": "NOOP", "target_memory_id": "", "updated_content": null}

---
## Instructions for Decision Making

1.  **Strict Comparison:** Carefully compare the 'Candidate Fact' with *each* 'Existing Similar Memory'. Pay close attention to the content and the similarity score.
2.  **Prioritize ADD:** If the 'Candidate Fact' brings **ANY genuinely new information or concept** that isn't fully captured or contradicted by an existing memory, even if there's slight topical overlap, the operation should lean towards **ADD**.
3.  **Specific Operations:**
    * If **UPDATE** or **DELETE**, you *must* identify the `target_memory_id` of the specific memory to be acted upon.
    * For **UPDATE**, generate a new, `updated_content` string that thoughtfully combines the information from the `Candidate Fact` and the `target_memory` to create a more comprehensive single memory.
    * For **ADD** or **NOOP**, `target_memory_id` should be an empty string `""` and `updated_content` should be `null`.
4.  **Adhere to Format:** Provide your response **ONLY as a JSON object**. Do not include any other text, explanations, or conversational fillers outside the JSON.

---


        """
        return prompt

def create_summary_prompt(memories):
    memories = "You are a summary writer that generates short and concise summary from existing memories"
    memories += "\n Memories start \n"
    for memory in memories:
        memories += f"{memory['content']}\n"
    memories += "\n Memories End \n"
    memories += """
    Note:
    1. Summary must be short in a paragraph only with only important facts that must be remembered.
    2. Avoid long sentences no matter if it is not gramatically correct\n
"""
    prompt = (
        "Generate a concise summary of the following conversation memories:\n"
        f"{memories}\n\n"
        "Summary:"
    )