import json
import faiss
import numpy as np
import requests
import os

class Database:
    def __init__(self,summary_file='./summary.txt', messages_file="./message.json",memories="./memories.json"):
        self.summary_file = summary_file
        self.messages_file = messages_file
        self.memories_file = memories
        self.conversation_summary = ""
        self.memories = []
        self.recent_messages = {}
        self.vector_index = None
        self.memory_embeddings = {}
        self.load_files()

    def load_files(self):
        with open(self.summary_file, 'r') as f:
            self.conversation_summary = f.read().strip()

        
        with open(self.messages_file, 'r') as f:
            messages_data = json.load(f)
            self.recent_messages = messages_data
        with open(self.memories_file, 'r') as f:
            memories_data = json.load(f)
            self.memories = memories_data

    def save_summary(self):
        """
        Save the conversation summary to a file.
        """
        with open(self.summary_file, 'w') as f:
            f.write(self.conversation_summary)

    def get_recent_messages(self, count=5):
        """
        Get recent messages from the loaded messages.
        """
        # Handle both list format (legacy) and dict format (new)
        if isinstance(self.recent_messages, list):
            return self.recent_messages[-count:]
        elif isinstance(self.recent_messages, dict) and "messages" in self.recent_messages:
            return self.recent_messages["messages"][-count:]
        else:
            return []

    def embed_text(self, text: str, model: str = "nomic-embed-text", ollama_url: str = "http://localhost:11434"):
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={
                "model": model,
                "prompt": text
            }
        )
        response.raise_for_status()
        embedding_data = response.json()
        return np.array(embedding_data["embedding"], dtype=np.float32)

    def create_vector_database(self, dimension=768,memory_file: str = "memory_embeddings.json",vector_index_file: str = "memory_index.faiss"):
        print("Creating vector database from memories...")
        
        # Initialize FAISS index
        self.vector_index = faiss.IndexFlatL2(dimension)
        embeddings_list = []
        memory_ids = []
        
        if os.path.exists(vector_index_file) and os.path.exists(memory_file):
            # Load existing vector index
            self.vector_index = faiss.read_index(vector_index_file)
            # Load existing memory embeddings
            with open(memory_file, 'r') as f:
                self.memory_embeddings = json.load(f)
            print("Loaded existing vector index and memory embeddings.")
            return self.vector_index
        else:
        # Generate embeddings for each memory
            for memory in self.memories:
                memory_id = memory.get('memory_id')
                content = memory.get('content')
                
                if memory_id and content:
                    embedding = self.embed_text(content)
                    embeddings_list.append(embedding)
                    memory_ids.append(memory_id)
                    self.memory_embeddings[memory_id] = {
                        'content': content,
                        'index_position': len(embeddings_list) - 1
                    }
                    print(f"Added embedding for {memory_id}")
        
        # Add all embeddings to FAISS index
            if embeddings_list:
                embeddings_matrix = np.vstack(embeddings_list)
                self.vector_index.add(embeddings_matrix)
        
        print(f"Vector database created with {len(embeddings_list)} memories")
        # Save the vector index to a file
        faiss.write_index(self.vector_index, "memory_index.faiss")
        # Save memory embeddings to a file
        with open("memory_embeddings.json", 'w') as f:
            json.dump(self.memory_embeddings, f, indent=2)
            
        return self.vector_index

    def similarity_search(self, query: str, k: int = 5):
        if self.vector_index is None:
            self.create_vector_database()
        
        query_embedding = self.embed_text(query).reshape(1, -1)
        distances, indices = self.vector_index.search(query_embedding, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.memory_embeddings):
                memory_data = list(self.memory_embeddings.values())[idx]
                memory_id = list(self.memory_embeddings.keys())[idx]
                results.append({
                    'memory_id': memory_id,
                    'content': memory_data['content'],
                    'score': 1.0 / (1.0 + distance),
                    'distance': distance
                })
        
        return results

    def _get_next_memory_id(self):
        if not self.memories:
            return "mem_001"
        
        existing_ids = [int(mem['memory_id'].split('_')[1]) for mem in self.memories if mem['memory_id'].startswith('mem_')]
        next_id = max(existing_ids) + 1
        return f"mem_{next_id:03d}"

    def _save_memories_to_file(self):
        with open(self.memories_file, 'w') as f:
            json.dump(self.memories, f, indent=2)

    def _rebuild_vector_index(self):
        if self.vector_index is not None:
            self.create_vector_database()

    def add_memory(self, content: str, updated_date: str = None):
        if updated_date is None:
            from datetime import datetime
            updated_date = datetime.now().isoformat()
        
        memory_id = self._get_next_memory_id()
        
        new_memory = {
            "memory_id": memory_id,
            "updated_date": updated_date,
            "content": content
        }
        
        self.memories.append(new_memory)
        self._save_memories_to_file()
        
        if self.vector_index is not None:
            embedding = self.embed_text(content)
            self.vector_index.add(embedding.reshape(1, -1))
            self.memory_embeddings[memory_id] = {
                'content': content,
                'index_position': len(self.memory_embeddings)
            }
            

    def update_memory(self, memory_id: str, new_content: str, updated_date: str = None):
        if updated_date is None:
            from datetime import datetime
            updated_date = datetime.now().isoformat()
        
        for memory in self.memories:
            if memory['memory_id'] == memory_id:
                memory['content'] = new_content
                memory['updated_date'] = updated_date
                break
        
        self._save_memories_to_file()
        
        if memory_id in self.memory_embeddings:
            new_embedding = self.embed_text(new_content)
            position = self.memory_embeddings[memory_id]['index_position']
            
            all_embeddings = []
            for i in range(self.vector_index.ntotal):
                if i == position:
                    all_embeddings.append(new_embedding)
                else:
                    old_embedding = self.vector_index.reconstruct(i)
                    all_embeddings.append(old_embedding)
            
            self.vector_index = faiss.IndexFlatL2(len(new_embedding))
            embeddings_matrix = np.vstack(all_embeddings)
            self.vector_index.add(embeddings_matrix)
            
            self.memory_embeddings[memory_id]['content'] = new_content

    def delete_memory(self, memory_id: str):
        if memory_id not in self.memory_embeddings:
            return
        
        delete_position = self.memory_embeddings[memory_id]['index_position']
        
        self.memories = [memory for memory in self.memories if memory['memory_id'] != memory_id]
        self._save_memories_to_file()
        
        all_embeddings = []
        new_memory_embeddings = {}
        new_position = 0
        
        for old_position in range(self.vector_index.ntotal):
            if old_position != delete_position:
                old_embedding = self.vector_index.reconstruct(old_position)
                all_embeddings.append(old_embedding)
                
                for mem_id, mem_data in self.memory_embeddings.items():
                    if mem_data['index_position'] == old_position:
                        new_memory_embeddings[mem_id] = {
                            'content': mem_data['content'],
                            'index_position': new_position
                        }
                        new_position += 1
                        break
        
        if all_embeddings:
            self.vector_index = faiss.IndexFlatL2(len(all_embeddings[0]))
            embeddings_matrix = np.vstack(all_embeddings)
            self.vector_index.add(embeddings_matrix)
        else:
            self.vector_index = faiss.IndexFlatL2(768)
        
        self.memory_embeddings = new_memory_embeddings

if __name__ == "__main__":
    db = Database()
    print("Recent Messages:", len(db.recent_messages))
    print("Memories:", len(db.memories))
    
    # Create vector database
    db.create_vector_database()
    
    # Test similarity search
    results = db.similarity_search("diabetes medication", k=3)
    print("\nSimilarity search results:")
    for result in results:
        print(f"ID: {result['memory_id']}, Score: {result['score']:.3f}")
        print(f"Content: {result['content']}")
        print()

    # Add a new memory
    db.add_memory("Patient has a history of asthma and uses an inhaler as needed.")
    
    # Test similarity search again
    results = db.similarity_search("asthma management", k=3)
    print("\nSimilarity search results after adding new memory:")
    for result in results:
        print(f"ID: {result['memory_id']}, Score: {result['score']:.3f}")
        print(f"Content: {result['content']}")

    # delete a memory
    db.delete_memory("mem_001")

    #save vector index
    db._rebuild_vector_index()