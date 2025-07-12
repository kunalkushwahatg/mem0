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

# run the model in other terminal
ollama run qwen2.5:3b-instruct
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
6. **Prompts** (`prompts.py`): Centralized prompt templates for consistency

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
