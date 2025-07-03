import requests
from langchain_community.chat_models import ChatOllama

class OllamaLLM:
    """LangChain-based wrapper for Ollama to work with the extraction system"""
    
    def __init__(self, model_name="qwen2:7b", temperature=0.3, ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.temperature = temperature
        self.ollama_url = ollama_url
        
        # Initialize LangChain ChatOllama
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=ollama_url
        )
    
    def predict(self, prompt):
        """Compatible with extraction.py expectations"""
        return self.generate(prompt)
    
    def generate(self, prompt, temperature=None, max_tokens=500):
        """Generate response using LangChain ChatOllama"""
        try:
            # Use instance temperature if not provided
            temp = temperature if temperature is not None else self.temperature
            
            # Update temperature if different from instance
            if temp != self.temperature:
                self.llm.temperature = temp
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            # Reset temperature if it was changed
            if temp != self.temperature:
                self.llm.temperature = self.temperature
                
            return response.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    def check_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self):
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
            return []
        except:
            return []
    
    def embed_text(self, text, model="nomic-embed-text"):
        """Generate embeddings using Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            embedding_data = response.json()
            return embedding_data["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    ollama_llm = OllamaLLM(model_name="qwen2.5:3b-instruct")
    
    if ollama_llm.check_connection():
        print("Ollama is running and accessible.")
        print("Available models:", ollama_llm.list_models())
        
        prompt = "What is the capital of France?"
        response = ollama_llm.predict(prompt)
        print("Response:", response)
    else:
        print("Ollama is not running or not accessible. Please start it with 'ollama serve'.")