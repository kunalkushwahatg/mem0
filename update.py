import json
import re
from typing import List, Dict, Tuple
from enum import Enum
from prompts import create_update_prompt
class MemoryOperation(Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"

class UpdatePhase:
    """
    Represents the Update Phase of the Mem0 system.
    
    This phase processes candidate facts from the Extraction Phase and determines
    appropriate memory management operations (ADD, UPDATE, DELETE, NOOP) through
    LLM-based reasoning and semantic similarity matching.
    """
    
    def __init__(self, llm, database, top_k_similar: int = 5):
        """
        Args:
            llm: LLM instance for decision making (with tool/function calling capability)
            database: Database interface for memory CRUD operations
            top_k_similar: Number of similar memories to retrieve for comparison
        """
        self.llm = llm
        self.database = database
        self.top_k_similar = top_k_similar
    
    def retrieve_similar_memories(self, candidate_fact: str) -> List[Dict]:
        similar_memories = self.database.similarity_search(
            query=candidate_fact,
            k=self.top_k_similar
        )
        return similar_memories
    
    
    
    def extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from LLM response, handling various formats"""
        try:
            # First try direct parsing
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Try to find JSON in the response using regex
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            # If no valid JSON found, return default ADD operation
            print(f"⚠️ Could not parse JSON from response: {response}")
            return {
                "operation": "ADD",
                "target_memory_id": "",
                "updated_content": None
            }
    
    def llm_decision_tool_call(self, candidate_fact: str, similar_memories: List[Dict]) -> Dict:
        prompt = create_update_prompt(candidate_fact, similar_memories)
        
        # Call LLM
        llm_response = self.llm.predict(prompt)
        print(f"LLM Response: {llm_response}")
        
        # Extract and parse JSON
        decision = self.extract_json_from_response(llm_response)
        
        # Validate the decision
        valid_operations = ["ADD", "UPDATE", "DELETE", "NOOP"]
        if decision.get("operation") not in valid_operations:
            print(f"⚠️ Invalid operation: {decision.get('operation')}, defaulting to ADD")
            decision["operation"] = "ADD"
        
        return decision
    
    def execute_operation(self, operation_decision: Dict, candidate_fact: str) -> bool:
        operation = operation_decision.get("operation")
        target_memory_id = operation_decision.get("target_memory_id")
        updated_content = operation_decision.get("updated_content")
        
        try:
            if operation == "ADD":
                self.database.add_memory(candidate_fact)
                print(f"✅ Added new memory: {candidate_fact[:50]}...")
            elif operation == "UPDATE":
                if target_memory_id and updated_content:
                    self.database.update_memory(target_memory_id, updated_content)
                    print(f"✅ Updated memory {target_memory_id}")
                else:
                    print(f"⚠️ UPDATE operation missing target_memory_id or updated_content, adding as new memory")
                    self.database.add_memory(candidate_fact)
            elif operation == "DELETE":
                if target_memory_id:
                    self.database.delete_memory(target_memory_id)
                    print(f"✅ Deleted memory {target_memory_id}")
                else:
                    print(f"⚠️ DELETE operation missing target_memory_id")
                    return False
            elif operation == "NOOP":
                print(f"ℹ️ No operation needed for: {candidate_fact[:50]}...")
            
            return True if operation in ["ADD", "UPDATE", "DELETE"] else False
            
        except Exception as e:
            print(f"❌ Error executing {operation} operation: {e}")
            return False
    
    def process_extracted_memories(self, extracted_memories: List[str]) -> List[Dict]:
        """Process a list of extracted memories and determine operations for each"""
        results = []
        
        for candidate_fact in extracted_memories:
            print(f"\n🔄 Processing candidate fact: {candidate_fact}")
            print("-" * 50)
            
            # Retrieve similar memories
            similar_memories = self.retrieve_similar_memories(candidate_fact)
            print(f"Found {len(similar_memories)} similar memories")
            
            # Get LLM decision
            operation_decision = self.llm_decision_tool_call(candidate_fact, similar_memories)
            print(f"LLM Decision: {operation_decision}")
            
            # Execute the operation
            success = self.execute_operation(operation_decision, candidate_fact)

            # Track results
            result = {
                "candidate_fact": candidate_fact,
                "operation_decision": operation_decision,
                "similar_memories_count": len(similar_memories),
                "execution_success": success
            }
            results.append(result)
            
            print(f"✅ Processed: {candidate_fact[:50]}... -> {operation_decision['operation']}")
        
        return results

# Example usage:
if __name__ == "__main__":
    from extraction import Extraction
    from database import Database
    from ollama_wrapper import OllamaLLM
    
    # Initialize components
    db = Database()
    llm = OllamaLLM(model_name="qwen2.5:3b-instruct", temperature=0.3)
    extractor = Extraction(llm, db)

    conversation_id = "12345"

    mt = "I visited taj mahal last year what where is it at?"
    mt_1 = "taj mahal is in agra"
    print("🔄 Extracting memories from conversation...")
    memories = extractor.extract_memories( mt_1, mt)
    print(f"📚 Extracted Memories: {memories}")
    if memories == []:
        print("No new memories extracted.")
        exit(0)
    # Initialize UpdatePhase with LLM and database
    print("\n🔄 Processing extracted memories...")
    update_phase = UpdatePhase(llm, db)
    results = update_phase.process_extracted_memories(memories)
    
    # Print summary
    print("\n📊 Processing Summary:")
    print("=" * 50)
    for result in results:
        print(f"Fact: {result['candidate_fact'][:50]}...")
        print(f"Operation: {result['operation_decision']['operation']}")
        print(f"Success: {result['execution_success']}")
        print("-" * 30)
