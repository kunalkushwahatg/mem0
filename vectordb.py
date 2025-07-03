import faiss
import numpy as np
import json
import pickle
import requests
import os
from typing import List, Dict, Optional, Tuple

class VectorDB:    
    def __init__(self, embedding_model: str = "nomic-embed-text", dimension: int = 768, index_file: str = "memory_index.faiss", metadata_file: str = "memory_metadata.pkl", ollama_url: str = "http://localhost:11434"):
        self.embedding_model_name = embedding_model
        self.dimension = dimension
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.ollama_url = ollama_url
        self.index = faiss.IndexFlatL2(dimension)
        self.memory_metadata = {}
        self.next_memory_id = 1
        self.load_index()
    
    def embed_text(self, text: str) -> np.ndarray:
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={
                "model": self.embedding_model_name,
                "prompt": text
            }
        )
        response.raise_for_status()
        embedding_data = response.json()
        embedding = np.array(embedding_data["embedding"], dtype=np.float32)
        return embedding.reshape(1, -1)
    

    def add_memory(self, memory_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        embedding = self.embed_text(content)
        self.index.add(embedding)
        memory_data = {
            'content': content,
            'index_position': self.index.ntotal - 1,
            'memory_id': memory_id
        }
        self.memory_metadata[memory_id] = memory_data
        print(f"Added memory {memory_id}: {content[:50]}...")
        return True
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        query_embedding = self.embed_text(query)
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            memory_id, memory_data = self._find_memory_by_position(idx)
            if memory_id:
                similarity_score = 1 / (1 + distance)
                result = {
                    'memory_id': memory_id,
                    'content': memory_data['content'],
                    'score': float(similarity_score),
                    'distance': float(distance)
                }
                results.append(result)
        return results
    
    def _find_memory_by_position(self, position: int) -> Tuple[Optional[str], Optional[Dict]]:
        for memory_id, memory_data in self.memory_metadata.items():
            if memory_data['index_position'] == position:
                return memory_id, memory_data
        return None, None
    
    def save_index(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file.replace('.pkl', '.json'), 'w') as f:
            json.dump(self.memory_metadata, f, indent=2)
        print(f"Saved index to {self.index_file} and metadata to {self.metadata_file}")

    def load_index(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            print(f"Loaded FAISS index from {self.index_file}")
        
        json_file = self.metadata_file.replace('.pkl', '.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                self.memory_metadata = json.load(f)
            print(f"Loaded metadata from {json_file}")
    
    def get_memory_count(self) -> int:
        return len(self.memory_metadata)
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict]:
        return self.memory_metadata.get(memory_id)

if __name__ == "__main__":
    import os
    vector_db = VectorDB(
        embedding_model="nomic-embed-text",
        dimension=768,
        ollama_url="http://localhost:11434"
    )
    vector_db.add_memory("mem_001", "Patient has diabetes and takes metformin daily")
    vector_db.add_memory("mem_002", "Patient exercises regularly and follows low-carb diet")
    vector_db.add_memory("mem_003", "Patient has hypertension controlled with medication")
    results = vector_db.similarity_search("diabetes management", k=3)
    print("\nSimilarity search results:")
    for result in results:
        print(f"ID: {result['memory_id']}, Score: {result['score']:.3f}")
        print(f"Content: {result['content']}")
        print()
    vector_db.save_index()
    print(f"Total memories: {vector_db.get_memory_count()}")