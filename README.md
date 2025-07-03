# Memory-Aware Chatbot with Ollama and Mem0

A sophisticated chatbot that combines Ollama's local LLM capabilities with a custom memory system (mem0) for persistent, context-aware conversations. Features advanced memory management with extraction, update, and retrieval phases powered by LangChain and vector similarity search.

## Features

- 🧠 **Persistent Memory**: Remembers information from previous conversations with intelligent storage
- 🔍 **Semantic Search**: Finds relevant memories using FAISS vector similarity search
- 💬 **Natural Conversations**: Context-aware responses using retrieved memories and conversation history
- 📚 **Advanced Memory Management**: Intelligent ADD, UPDATE, DELETE, and NOOP operations
- 🤖 **Local LLM**: Uses Ollama with LangChain for privacy-focused AI interactions
- ⚡ **Multi-Phase Architecture**: Extraction → Update → Retrieval pipeline for memory management
- 🔄 **Automatic Memory Extraction**: LLM-powered extraction of important facts from conversations
- 📊 **Memory Analytics**: Search, view, and manage stored memories with similarity scoring

## Prerequisites

1. **Python 3.8+**
2. **Ollama**: Install from [https://ollama.ai](https://ollama.ai)
3. **LangChain**: For LLM integration (included in requirements)
4. **FAISS**: For vector similarity search (included in requirements)

## Quick Start

### 1. Setup Ollama

```powershell
# Install and start Ollama (if not already done)
ollama serve

# In another terminal, pull a model
ollama pull qwen2.5:3b-instruct
```

### 2. Install Dependencies

```powershell
# Install Python dependencies
pip install -r requirements.txt
```

Or run the setup script for automated checking:
```powershell
python setup.py
```

### 3. Run the Chatbot

```powershell
python chat.py
```

## Usage

### Basic Chat
Just type naturally and the chatbot will respond while learning about you:

```
👤 You: Hi, I'm working on a React project
🤖 Assistant: Hello! That's great that you're working on a React project...
```

### Special Commands

- `memories` - View recently stored memories with details
- `search: <query>` - Search memories using semantic similarity
- `help` - Show all available commands
- `quit` / `exit` / `bye` - End the conversation gracefully

### Advanced Features

The chatbot automatically:
- **Extracts memories**: Identifies important facts from conversations
- **Updates existing memories**: Merges or updates related information
- **Removes redundant data**: Avoids storing duplicate information
- **Maintains context**: Uses recent conversation history for better responses

### Example Session

```
👤 You: I love playing Valorant, especially as Sage
🤖 Assistant: That's awesome! Sage is a great agent...

👤 You: memories
📚 Recent Memories:
- User enjoys playing Valorant and prefers playing as Sage agent

👤 You: search: gaming
🔍 Searching memories for: 'gaming'
1. Score: 0.892
   Content: User enjoys playing Valorant and prefers playing as Sage agent
```

## Architecture

### Memory Management Pipeline

The system follows a sophisticated 3-phase architecture:

1. **Extraction Phase** (`extraction.py`): Analyzes conversations and extracts important facts
2. **Update Phase** (`update.py`): Determines appropriate memory operations (ADD/UPDATE/DELETE/NOOP)
3. **Retrieval Phase**: Searches and retrieves relevant memories for context

### Components

1. **Database** (`database.py`): Handles memory storage, vector operations, and FAISS indexing
2. **Extraction** (`extraction.py`): LLM-powered extraction of important information from conversations
3. **Update** (`update.py`): Intelligent memory management with conflict resolution
4. **OllamaLLM** (`ollama_wrapper.py`): LangChain-based wrapper for Ollama API integration
5. **MemoryAwareChatbot** (`chat.py`): Main orchestrator coordinating all components
6. **VectorDB** (`vectordb.py`): Specialized vector database utilities
7. **Prompts** (`prompts.py`): Centralized prompt templates for consistency

### Memory System

- **Storage**: JSON files for memories and conversation history with structured metadata
- **Vector DB**: FAISS for high-performance semantic similarity search
- **Embeddings**: Uses Ollama's embedding models (nomic-embed-text) for vector representations
- **Extraction**: LLM-powered fact extraction with context-aware prompting
- **Updates**: Intelligent memory operations to prevent redundancy and maintain accuracy
- **Conflict Resolution**: Automatic handling of contradictory or duplicate information

## File Structure

```
mem0/
├── chat.py              # Main chatbot application with interactive loop
├── ollama_wrapper.py    # LangChain-based Ollama API wrapper
├── database.py          # Memory storage and FAISS vector operations
├── extraction.py        # Memory extraction logic with context assembly
├── update.py            # Memory update phase with intelligent operations
├── vectordb.py          # Vector database utilities and operations
├── prompts.py           # Centralized prompt templates
├── memories.json        # Stored memories with metadata
├── message.json         # Conversation history and context
├── summary.txt          # Conversation summary for context
├── memory_embeddings.json # Memory embeddings cache
├── memory_index.faiss   # FAISS vector index for similarity search
├── requirements.txt     # Python dependencies including LangChain
├── setup.py            # Setup script with dependency checking
└── README.md           # This file
```

## Configuration

### Changing Models

Edit the model name in `chat.py`:

```python
chatbot = MemoryAwareChatbot(model_name="mistral")  # or any other Ollama model
```

### Adjusting Memory Settings

In `MemoryAwareChatbot.__init__()`:
- Modify similarity thresholds for memory retrieval
- Change context window sizes for conversation history
- Adjust memory extraction sensitivity and filtering
- Configure vector database parameters (dimensions, similarity metrics)

### Memory Operations

The system supports four types of memory operations:
- **ADD**: Store completely new information
- **UPDATE**: Enhance existing memories with additional details
- **DELETE**: Remove outdated or incorrect information
- **NOOP**: No operation needed (information already exists or irrelevant)

## Troubleshooting

### Ollama Connection Issues
```
❌ Error generating response: Connection refused
```
**Solution**: Make sure Ollama is running with `ollama serve`

### Model Not Found
```
❌ Error: model 'qwen2.5:3b-instruct' not found
```
**Solution**: Pull the model with `ollama pull qwen2.5:3b-instruct`

### LangChain Import Issues
```
❌ ImportError: No module named 'langchain_community'
```
**Solution**: Install LangChain dependencies with `pip install -r requirements.txt`

### JSON Parsing Errors
```
❌ JSONDecodeError: Expecting value
```
**Solution**: The system includes robust JSON parsing with fallbacks. Check LLM responses and prompt formatting.

### Vector Index Issues
```
❌ Error loading vector index
```
**Solution**: Delete `memory_index.faiss` and `memory_embeddings.json` to rebuild the vector database.

### Memory Loading Issues
```
❌ Error loading memories
```
**Solution**: Check if JSON files exist and are valid. Delete corrupted files to reset.

## Advanced Usage

### Memory Pipeline Testing

```python
from extraction import Extraction
from update import UpdatePhase
from database import Database
from ollama_wrapper import OllamaLLM

# Initialize components
db = Database()
llm = OllamaLLM(model_name="qwen2.5:3b-instruct", temperature=0.3)

# Test extraction
extractor = Extraction(llm, db)
memories = extractor.extract_memories("conv_id", "user message", "assistant response")

# Test update phase
updater = UpdatePhase(llm, db)
results = updater.process_extracted_memories(memories)
```

### Custom Memory Addition

```python
from database import Database

db = Database()
db.add_memory("User prefers dark theme in applications")
```

### Advanced Memory Search

```python
# Search with custom parameters
results = db.similarity_search("user interface preferences", k=10)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content']}")
    print(f"Memory ID: {result['memory_id']}")
```

### Batch Memory Operations

```python
# Batch import with automatic update detection
from update import UpdatePhase

memories = [
    "User is a software developer",
    "User prefers Python for data science", 
    "User works remotely from home"
]

updater = UpdatePhase(llm, db)
results = updater.process_extracted_memories(memories)
```

### Custom LLM Configuration

```python
# Use different models for different tasks
chat_llm = OllamaLLM(model_name="qwen2.5:7b-instruct", temperature=0.7)
extraction_llm = OllamaLLM(model_name="qwen2.5:3b-instruct", temperature=0.3)

chatbot = MemoryAwareChatbot(model_name="qwen2.5:7b-instruct")
```

## Performance & Scalability

- **Memory Efficiency**: FAISS indexing for fast similarity search even with large memory stores
- **Incremental Updates**: Only processes new memories without rebuilding entire index
- **Configurable Parameters**: Adjustable similarity thresholds and context window sizes
- **Error Recovery**: Robust error handling with graceful fallbacks
- **Model Flexibility**: Support for different Ollama models with varying capabilities

## Dependencies

Current requirements include:
- `requests>=2.31.0` - HTTP requests for Ollama API
- `numpy>=1.24.0` - Numerical computations for embeddings
- `faiss-cpu>=1.7.4` - Vector similarity search
- `langchain>=0.1.0` - LLM framework integration
- `langchain-community>=0.0.10` - Community LLM providers

## Contributing

Feel free to submit issues and enhancement requests! Areas for contribution:
- Additional LLM providers support
- Enhanced memory extraction algorithms
- Performance optimizations
- UI/Web interface development
- Testing and documentation improvements

## License

This project is open source and available under the MIT License.
