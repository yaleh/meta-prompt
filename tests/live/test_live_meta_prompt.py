#!/usr/bin/env python3
"""
Live test script for the meta-prompt process with real LLM servers.
This script demonstrates the complete meta-prompt workflow using actual LLM APIs.
"""

import json
import logging
from pathlib import Path
import yaml
from langchain_openai import ChatOpenAI

from meta_prompt import MetaPromptGraph, AgentState, Example
from meta_prompt.consts import META_PROMPT_NODES

# Configure logging to see the workflow in action
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_config():
    """Load configuration from config.yml"""
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_llm_from_config(llm_config):
    """Create ChatOpenAI instance from config"""
    return ChatOpenAI(
        model_name=llm_config['model_name'],
        openai_api_key=llm_config['openai_api_key'],
        openai_api_base=llm_config['openai_api_base'],
        temperature=llm_config['temperature'],
        max_tokens=llm_config['max_tokens'],
        verbose=llm_config.get('verbose', False),
        max_retries=3
    )

def test_simple_meta_prompt():
    """Test a simple meta-prompt scenario"""
    print("🚀 Starting Meta-Prompt Live Test")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Use a smaller model for faster testing
    model_name = "llama3-8b-8192"  # Faster model for testing
    if model_name not in config['llms']:
        print(f"❌ Model {model_name} not found in config")
        return
    
    llm_config = config['llms'][model_name]
    print(f"📡 Using model: {model_name}")
    print(f"🔗 API endpoint: {llm_config['openai_api_base']}")
    
    # Create LLM instances for all nodes
    llm = create_llm_from_config(llm_config)
    llms = {node: llm for node in META_PROMPT_NODES}
    
    # Create MetaPromptGraph
    graph = MetaPromptGraph(llms=llms)
    
    # Define test scenario
    test_example = Example(
        user_message="How do I reverse a list in Python?",
        expected_output="You can reverse a list in Python using the reverse() method: my_list.reverse(). This modifies the list in place."
    )
    
    input_state = AgentState(
        examples=[test_example],
        acceptance_criteria="The response should mention the reverse() method and explain that it modifies the list in place.",
        max_output_age=2  # Keep it small for faster testing
    )
    
    print("\n📝 Test Scenario:")
    print(f"   User Message: {test_example['user_message']}")
    print(f"   Expected Output: {test_example['expected_output']}")
    print(f"   Acceptance Criteria: {input_state['acceptance_criteria']}")
    print(f"   Max Output Age: {input_state['max_output_age']}")
    
    try:
        print("\n🔄 Running Meta-Prompt Workflow...")
        print("-" * 40)
        
        # Run the meta-prompt process
        result = graph.run_meta_prompt_graph(input_state, recursion_limit=15)
        
        print("\n✅ Meta-Prompt Process Complete!")
        print("=" * 60)
        
        # Display results
        print("\n📊 RESULTS:")
        print("-" * 20)
        
        print(f"\n🎯 Accepted: {result.get('accepted', False)}")
        print(f"🔄 Best Output Age: {result.get('best_output_age', 'N/A')}")
        
        if result.get('best_system_message'):
            print(f"\n🤖 Generated System Message:")
            print(f"   {result['best_system_message']}")
        
        if result.get('best_output'):
            print(f"\n💬 Best Output:")
            print(f"   {result['best_output']}")
        
        if result.get('analysis'):
            print(f"\n📈 Final Analysis:")
            try:
                analysis = json.loads(result['analysis'])
                print(f"   Accept: {analysis.get('Accept', 'N/A')}")
                if analysis.get('Acceptable Differences'):
                    print(f"   Acceptable Differences: {analysis['Acceptable Differences']}")
                if analysis.get('Unacceptable Differences'):
                    print(f"   Unacceptable Differences: {analysis['Unacceptable Differences']}")
            except:
                print(f"   {result['analysis']}")
        
        # Test the generated system message
        if result.get('best_system_message'):
            print(f"\n🧪 Testing Generated System Message:")
            print("-" * 40)
            
            messages = [
                ("system", result['best_system_message']),
                ("human", "How do I sort a list in Python?")
            ]
            
            test_response = llm.invoke(messages)
            print(f"   Test Question: How do I sort a list in Python?")
            print(f"   Generated Response: {test_response.content}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during meta-prompt process: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def test_complex_meta_prompt():
    """Test a more complex meta-prompt scenario"""
    print("\n🚀 Starting Complex Meta-Prompt Test")
    print("=" * 60)
    
    config = load_config()
    model_name = "llama3-70b-8192"  # Use larger model for complex task
    
    if model_name not in config['llms']:
        print(f"❌ Model {model_name} not found in config")
        return
    
    llm_config = config['llms'][model_name]
    print(f"📡 Using model: {model_name}")
    
    llm = create_llm_from_config(llm_config)
    llms = {node: llm for node in META_PROMPT_NODES}
    
    graph = MetaPromptGraph(llms=llms)
    
    # More complex example requiring detailed explanation
    complex_example = Example(
        user_message="Explain how to implement a binary search algorithm",
        expected_output="Binary search is an efficient algorithm for finding an item in a sorted array. It works by repeatedly dividing the search interval in half. Here's how: 1) Compare the target with the middle element, 2) If they match, return the position, 3) If target is less than middle, search the left half, 4) If target is greater, search the right half, 5) Repeat until found or interval is empty. Time complexity is O(log n)."
    )
    
    input_state = AgentState(
        examples=[complex_example],
        acceptance_criteria="The response must explain the binary search algorithm step-by-step, mention the divide-and-conquer approach, and include the O(log n) time complexity.",
        max_output_age=3
    )
    
    print("\n📝 Complex Test Scenario:")
    print(f"   User Message: {complex_example['user_message']}")
    print(f"   Expected: [Complex algorithm explanation]")
    print(f"   Acceptance Criteria: {input_state['acceptance_criteria']}")
    
    try:
        print("\n🔄 Running Complex Meta-Prompt Workflow...")
        result = graph.run_meta_prompt_graph(input_state, recursion_limit=20)
        
        print("\n✅ Complex Meta-Prompt Complete!")
        print(f"🎯 Accepted: {result.get('accepted', False)}")
        
        if result.get('best_system_message'):
            print(f"\n🤖 Generated System Message:")
            print(f"   {result['best_system_message'][:200]}...")
        
        if result.get('best_output'):
            print(f"\n💬 Best Output:")
            print(f"   {result['best_output'][:300]}...")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None

def main():
    """Main test function"""
    print("🧪 Meta-Prompt Live Testing Suite")
    print("=" * 60)
    
    # Test 1: Simple scenario
    simple_result = test_simple_meta_prompt()
    
    if simple_result and simple_result.get('accepted'):
        print(f"\n✅ Simple test PASSED - System message was accepted!")
    else:
        print(f"\n⚠️  Simple test completed but may need more iterations")
    
    # Uncomment to run complex test (takes longer)
    # print("\n" + "="*60)
    # complex_result = test_complex_meta_prompt()
    
    print(f"\n🏁 Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()