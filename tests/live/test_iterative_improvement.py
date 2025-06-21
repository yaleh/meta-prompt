#!/usr/bin/env python3
"""
Test iterative improvement in the meta-prompt process.
This test uses a more demanding scenario that likely requires multiple iterations.
"""

import json
import logging
from pathlib import Path
import yaml
from langchain_openai import ChatOpenAI

from meta_prompt import MetaPromptGraph, AgentState, Example
from meta_prompt.consts import META_PROMPT_NODES

# Configure detailed logging to see the iterative process
logging.basicConfig(
    level=logging.DEBUG,
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

def test_demanding_scenario():
    """Test a scenario likely to require iterative improvement"""
    print("🔄 Testing Iterative Improvement Scenario")
    print("=" * 60)
    
    config = load_config()
    model_name = "llama3-70b-8192"  # Use larger model for better performance
    
    if model_name not in config['llms']:
        print(f"❌ Model {model_name} not found in config")
        return
    
    llm_config = config['llms'][model_name]
    print(f"📡 Using model: {model_name}")
    
    llm = create_llm_from_config(llm_config)
    llms = {node: llm for node in META_PROMPT_NODES}
    
    graph = MetaPromptGraph(llms=llms)
    
    # Demanding scenario: requires specific format and technical accuracy
    demanding_example = Example(
        user_message="Explain the difference between Python lists and tuples",
        expected_output="""Python lists and tuples are both sequence types, but they have key differences:

**Lists:**
- Mutable (can be modified after creation)
- Created with square brackets: [1, 2, 3]
- Support item assignment, append, remove operations
- Use more memory due to mutability overhead

**Tuples:**
- Immutable (cannot be modified after creation)
- Created with parentheses: (1, 2, 3)
- Do not support item assignment or modification
- More memory efficient and faster access
- Can be used as dictionary keys (hashable)

Example:
```python
my_list = [1, 2, 3]
my_list[0] = 10  # Works
my_tuple = (1, 2, 3)
my_tuple[0] = 10  # Error!
```"""
    )
    
    input_state = AgentState(
        examples=[demanding_example],
        acceptance_criteria="""The response must:
1. Clearly state that lists are mutable and tuples are immutable
2. Show the syntax difference with square brackets vs parentheses
3. Mention at least 3 practical differences (modification, memory, performance)
4. Include a code example demonstrating the mutability difference
5. Use proper markdown formatting with headers and code blocks""",
        max_output_age=4  # Allow more iterations
    )
    
    print("\n📝 Demanding Test Scenario:")
    print(f"   Topic: Python lists vs tuples comparison")
    print(f"   Requirements: Specific format, technical accuracy, code examples")
    print(f"   Max Iterations: {input_state['max_output_age']}")
    
    try:
        print("\n🔄 Running Iterative Meta-Prompt Process...")
        print("-" * 50)
        
        # Track iterations
        iteration_count = 0
        
        # Run with custom callback to track progress (if available)
        result = graph.run_meta_prompt_graph(input_state, recursion_limit=25)
        
        print("\n✅ Iterative Process Complete!")
        print("=" * 60)
        
        # Analyze the results
        print(f"\n📊 FINAL RESULTS:")
        print(f"🎯 Accepted: {result.get('accepted', False)}")
        print(f"🔄 Final Output Age: {result.get('best_output_age', 'N/A')}")
        print(f"📈 Total Iterations: {result.get('best_output_age', 0) + 1}")
        
        if result.get('best_system_message'):
            print(f"\n🤖 Final System Message:")
            print(f"   {result['best_system_message'][:300]}...")
        
        if result.get('best_output'):
            print(f"\n💬 Final Output:")
            print("-" * 30)
            output = result['best_output']
            # Show first 500 chars to see the structure
            print(output[:500])
            if len(output) > 500:
                print("... [truncated]")
        
        # Check if it meets our demanding criteria
        output = result.get('best_output', '')
        criteria_met = []
        
        if 'mutable' in output.lower() and 'immutable' in output.lower():
            criteria_met.append("✅ Mentions mutability difference")
        else:
            criteria_met.append("❌ Missing mutability explanation")
            
        if '[' in output and ']' in output and '(' in output and ')' in output:
            criteria_met.append("✅ Shows syntax difference")
        else:
            criteria_met.append("❌ Missing syntax examples")
            
        if 'memory' in output.lower() or 'performance' in output.lower():
            criteria_met.append("✅ Mentions performance/memory")
        else:
            criteria_met.append("❌ Missing performance discussion")
            
        if '```' in output or 'python' in output.lower():
            criteria_met.append("✅ Includes code examples")
        else:
            criteria_met.append("❌ Missing code examples")
            
        if '#' in output or '**' in output:
            criteria_met.append("✅ Uses markdown formatting")
        else:
            criteria_met.append("❌ Missing markdown formatting")
        
        print(f"\n📋 Criteria Analysis:")
        for criterion in criteria_met:
            print(f"   {criterion}")
        
        # Test the final system message with a related question
        if result.get('best_system_message'):
            print(f"\n🧪 Testing Final System Message:")
            print("-" * 40)
            
            messages = [
                ("system", result['best_system_message']),
                ("human", "What are the advantages of using tuples over lists?")
            ]
            
            test_response = llm.invoke(messages)
            print(f"   Test Q: What are the advantages of using tuples over lists?")
            print(f"   Response: {test_response.content[:200]}...")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during iterative process: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("🔄 Meta-Prompt Iterative Improvement Test")
    print("=" * 60)
    
    result = test_demanding_scenario()
    
    if result:
        if result.get('accepted'):
            print(f"\n🎉 SUCCESS: Meta-prompt achieved acceptance!")
            print(f"   Final output age: {result.get('best_output_age', 0)}")
            print(f"   Iterations needed: {result.get('best_output_age', 0) + 1}")
        else:
            print(f"\n⚠️  PARTIAL: Process completed but didn't reach acceptance")
            print(f"   Final output age: {result.get('best_output_age', 0)}")
            print(f"   May need higher max_output_age for full convergence")
    else:
        print(f"\n❌ FAILED: Process encountered errors")
    
    print(f"\n🏁 Iterative Test Complete!")

if __name__ == "__main__":
    main()