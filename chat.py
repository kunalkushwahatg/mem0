import json
import os
from datetime import datetime
from database import Database
from extraction import Extraction
from ollama_wrapper import OllamaLLM
from update import UpdatePhase
from prompts import create_chat_prompt
class MemoryAwareChatbot:
    """A chatbot that uses mem0 for memory management and Ollama for generation"""
    
    def __init__(self, model_name="qwen2:7b"):
        self.llm = OllamaLLM(model_name)
        

        if not self.llm.check_connection():
            print("⚠️  Warning: Cannot connect to Ollama. Make sure it's running with 'ollama serve'")
        
        self.db = Database()
        
        self.extractor = Extraction(self.llm, self.db)
        self.conversation_history = []
        self.update_phase = UpdatePhase(self.llm, self.db)

        
        
        # Initialize vector database if not exists
        if self.db.vector_index is None:
            print("Initializing vector database...")
            self.db.create_vector_database()
        
        print("✅ Chatbot initialized successfully!")
        print(f"📚 Loaded memories")
        print(f"🤖 Using model: {model_name}")
    
    def _save_message_to_history(self, user_message, bot_response):
        """Save the conversation turn to message history"""
        timestamp = datetime.now().isoformat()
        message_pair = {
            "timestamp": timestamp,
            "user": user_message,
            "assistant": bot_response
        }
        
        self.conversation_history.append(message_pair)
        
        # Handle both list format (legacy) and dict format (new)
        if isinstance(self.db.recent_messages, list):
            # Convert to new format
            self.db.recent_messages = {"messages": self.db.recent_messages}
        
        # Update the database's recent messages
        if "messages" not in self.db.recent_messages:
            self.db.recent_messages["messages"] = []
        
        self.db.recent_messages["messages"].append(message_pair)
        
        # Save to file
        with open(self.db.messages_file, 'w') as f:
            json.dump(self.db.recent_messages, f, indent=2)

    def _get_summary(self, user_message):
        """Retrieve summary in the database"""
        try:
            summary = self.db.conversation_summary

            if summary:
                context = "Relevant information from previous conversations:\n"
                context += f"{summary}\n"
                return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""

    def _get_recent_conversation(self, n=3):
        """Get recent conversation context"""
        if len(self.conversation_history) > 0:
            recent = self.conversation_history[-n:]
            context = "Recent conversation:\n"
            for turn in recent:
                context += f"User: {turn['user']}\n"
                context += f"Assistant: {turn['assistant']}\n\n"
            return context
        return ""
    
    def chat(self, user_message):
        """Main chat method"""
        print(f"👤 User: {user_message}")
        self.extractor.messages_count += 1
        
        # Get relevant context from memories
        memory_context = self._get_summary(user_message)

        # Get recent conversation context
        recent_context = self._get_recent_conversation()
        
        # Build the complete prompt
        full_prompt = create_chat_prompt( user_message, memory_context, recent_context)
        
        # print(f"🤖 Generating response with context:\n{full_prompt}")
        # Generate response
        try:
            response = self.llm.generate(full_prompt, temperature=0.7)
            print(f"🤖 Assistant: {response}")
            
            # Save the conversation
            self._save_message_to_history(user_message, response)
            
            # Extract and store memories
            memories = self.extractor.extract_memories(user_message, response)
            if memories == []:
                print("No new memories extracted.")
                return response
            self.update_phase.process_extracted_memories(memories)
            return response
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {e}"
            print(f"🤖 Assistant: {error_msg}")
            return error_msg
    
    def show_memories(self, limit=10):
        """Display stored memories"""
        print(f"\n📚 Recent Memories (showing last {limit}):")
        print("-" * 50)
        recent_memories = self.db.memories[-limit:]
        for memory in recent_memories:
            print(f"ID: {memory['memory_id']}")
            print(f"Content: {memory['content']}")
            print(f"Date: {memory['updated_date']}")
            print("-" * 30)
    
    def search_memories(self, query, k=5):
        """Search memories by similarity"""
        print(f"\n🔍 Searching memories for: '{query}'")
        print("-" * 50)
        results = self.db.similarity_search(query, k=k)
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Content: {result['content']}")
            print(f"   ID: {result['memory_id']}")
            print()

def main():
    """Main interactive chat loop"""
    print("🚀 Starting Memory-Aware Chatbot...")
    print("=" * 50)
    
    try:
        # Initialize chatbot
        chatbot = MemoryAwareChatbot(model_name="qwen2:7b")
        
        print("\n💬 Chat started! Type 'quit' to exit, 'memories' to view stored memories, or 'search: <query>' to search memories.")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! It was nice chatting with you.")
                    break
                
                elif user_input.lower() == 'memories':
                    chatbot.show_memories()
                    continue
                
                elif user_input.lower().startswith('search:'):
                    query = user_input[7:].strip()
                    if query:
                        chatbot.search_memories(query)
                    else:
                        print("Please provide a search query: search: <your query>")
                    continue
                
                elif user_input.lower() == 'help':
                    print("""
Available commands:
- quit/exit/bye: End the conversation
- memories: Show recent stored memories
- search: <query>: Search memories for specific content
- help: Show this help message
- Just type normally to chat!
                    """)
                    continue
                
                elif not user_input:
                    continue
                
                # Regular chat
                chatbot.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! It was nice chatting with you.")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                print("Let's continue chatting...")
    
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("Please make sure Ollama is running and you have the required model installed.")
        print("Run: ollama pull qwen2.5:3b-instruct")

if __name__ == "__main__":
    main()