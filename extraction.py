from prompts import form_extraction_prompt
from prompts import create_summary_prompt
class Extraction:
    """
    Represents the Extraction Phase of the Mem0 system.

    This phase is responsible for taking a new message pair, gathering
    relevant conversational context, and using an LLM to extract salient
    memories from the exchange.
    """

    def __init__(self, llm, db, recency_window_m: int = 2, update_summary_after: int = 10):
        """
        Args:
            llm: An Ollama-compatible LLM instance with a .predict(prompt) or .invoke(prompt) method.
            db: Database interface for fetching summaries and recent messages.
            recency_window_m: Number of recent messages to include as context.
        """
        self.llm = llm
        self.update_summary_after = update_summary_after
        self.messages_count = 0 
        self.db = db
        self.recency_window_m = recency_window_m

        self.generate_summary()

    def generate_summary(self):
        """
        Generates a summary of the conversation using the LLM.
        This method is not used in the current extraction workflow but can be implemented if needed.
        """
        # Takes memories content and passes to LLM to generate a summary
        prompt = create_summary_prompt(self.db.memories)

        print(f"Generating Summary of past context please wait it takes time ......")
        summary = self.llm.predict(prompt) 

        self.db.conversation_summary = summary
        self.db.save_summary()
        print(f"Generated Summary: {summary}")

    def assemble_context(self):
        """
        Gathers the conversation summary and recent messages for context.
        """
        # Retrieve the most recent conversation summary (S)
        summary = self.db.conversation_summary
        # Retrieve the last m messages (excluding the current pair)
        recent_messages = self.db.get_recent_messages(self.recency_window_m)
        return summary, recent_messages
    

    def extract_memories(self, mt_1, mt):
        """
        Main extraction workflow for a new message pair.
        """

        summary, recent_messages = self.assemble_context()

        # Step 2: Form prompt
        prompt = form_extraction_prompt(summary, recent_messages, mt_1, mt)

        # Step 3: LLM extraction (Ollama model)
        # If your LLM is async, use: memories = await self.llm.invoke(prompt)
        memories = self.llm.predict(prompt)
        if "<none>" in memories:
            return []
        if self.messages_count >= self.update_summary_after:
            # Update the summary after a certain number of messages
            self.generate_summary()
            self.messages_count = 0

        if isinstance(memories, str):
            # Simple parsing: split by lines or bullets
            extracted = [line.strip("- ").strip() for line in memories.strip().splitlines() if line.strip()]
        else:
            extracted = memories

        return extracted


if __name__ == "__main__":
    from database import Database
    from ollama_wrapper import OllamaLLM

    db = Database()
    llm = OllamaLLM(model_name="qwen2:7b", temperature=0.3)
    extractor = Extraction(llm, db)
    mt = "Hi how are you doing today?"
    mt_1 = "hello i am doing great"
    
    print(extractor.extract_memories(mt_1,mt))