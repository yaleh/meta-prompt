#!/usr/bin/env python3
"""
Simple demo of the meta-prompt process working successfully.
"""

import json
from pathlib import Path
import yaml
from langchain_openai import ChatOpenAI

from meta_prompt import MetaPromptGraph, AgentState, Example
from meta_prompt.consts import META_PROMPT_NODES

def load_config():
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_llm_from_config(llm_config):
    return ChatOpenAI(
        model_name=llm_config['model_name'],
        openai_api_key=llm_config['openai_api_key'],
        openai_api_base=llm_config['openai_api_base'],
        temperature=llm_config['temperature'],
        max_tokens=llm_config['max_tokens'],
        verbose=llm_config.get('verbose', False),
        max_retries=3
    )

def test_basic_workflow():
    """Test basic meta-prompt workflow"""
    print("🧪 Simple Meta-Prompt Demo")
    print("=" * 50)
    
    config = load_config()
    model_name = "llama3-8b-8192"  # Use smaller, faster model
    
    llm_config = config['llms'][model_name]
    llm = create_llm_from_config(llm_config)
    llms = {node: llm for node in META_PROMPT_NODES}
    
    graph = MetaPromptGraph(llms=llms)
    
    # Simple, clear example
    example = Example(
        user_message="What is the capital of France?",
        expected_output="The capital of France is Paris."
    )
    
    input_state = AgentState(
        examples=[example],
        acceptance_criteria="Response should clearly state that Paris is the capital of France.",
        max_output_age=1  # Keep very simple
    )
    
    print(f"📝 Test Case:")
    print(f"   Question: {example['user_message']}")
    print(f"   Expected: {example['expected_output']}")
    print(f"   Criteria: {input_state['acceptance_criteria']}")
    
    try:
        print(f"\n🔄 Running meta-prompt process...")
        result = graph.run_meta_prompt_graph(input_state, recursion_limit=10)
        
        print(f"\n✅ RESULTS:")
        print(f"   ✓ Accepted: {result.get('accepted', False)}")
        print(f"   ✓ Output Age: {result.get('best_output_age', 'N/A')}")
        
        if result.get('best_system_message'):
            print(f"\n🤖 Generated System Message:")
            print(f"   {result['best_system_message'][:150]}...")
        
        if result.get('best_output'):
            print(f"\n💬 Generated Answer:")
            print(f"   {result['best_output']}")
        
        # Quick test of the system message
        if result.get('best_system_message'):
            messages = [
                ("system", result['best_system_message']),
                ("human", "What is the capital of Italy?")
            ]
            test_response = llm.invoke(messages)
            print(f"\n🧪 Test with similar question:")
            print(f"   Q: What is the capital of Italy?")
            print(f"   A: {test_response.content}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_basic_workflow()
    if success:
        print(f"\n🎉 Meta-prompt process completed successfully!")
    else:
        print(f"\n⚠️  Process encountered issues")