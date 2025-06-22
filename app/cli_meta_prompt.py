#!/usr/bin/env python3
"""
Command Line Interface for Meta Prompt
Provides a CLI for generating and optimizing LLM prompts from examples.
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from confz import CLArgSource, EnvSource, FileSource
from app.config import MetaPromptConfig
from meta_prompt import MetaPromptGraph, AgentState, Example


class MetaPromptCLI:
    """Command line interface for Meta Prompt system."""
    
    def __init__(self, config_file: str = "config.yml"):
        """Initialize CLI with configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        self.graph = None
    
    def _load_config(self) -> MetaPromptConfig:
        """Load configuration from file and environment."""
        config_sources = [
            FileSource(file=self.config_file, optional=True),
            EnvSource(prefix='METAPROMPT_', allow_all=True),
            CLArgSource()
        ]
        return MetaPromptConfig(config_sources=config_sources)
    
    def _load_examples_from_file(self, examples_file: str) -> List[Example]:
        """Load examples from JSON or YAML file."""
        examples_path = Path(examples_file)
        if not examples_path.exists():
            raise FileNotFoundError(f"Examples file not found: {examples_file}")
        
        with open(examples_path, 'r', encoding='utf-8') as f:
            if examples_path.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        examples = []
        if isinstance(data, list):
            for item in data:
                examples.append(Example(
                    user_message=item.get('user_message', item.get('input', '')),
                    expected_output=item.get('expected_output', item.get('output', ''))
                ))
        elif isinstance(data, dict) and 'examples' in data:
            for item in data['examples']:
                examples.append(Example(
                    user_message=item.get('user_message', item.get('input', '')),
                    expected_output=item.get('expected_output', item.get('output', ''))
                ))
        else:
            raise ValueError("Invalid examples file format. Expected list of examples or dict with 'examples' key.")
        
        return examples
    
    def _create_examples_from_args(self, user_message: str, expected_output: str) -> List[Example]:
        """Create examples list from command line arguments."""
        return [Example(user_message=user_message, expected_output=expected_output)]
    
    def _load_examples(self, examples_file: Optional[str] = None, 
                      user_message: Optional[str] = None, 
                      expected_output: Optional[str] = None) -> List[Example]:
        """Load examples from file or command line arguments."""
        if examples_file:
            return self._load_examples_from_file(examples_file)
        elif user_message and expected_output:
            return self._create_examples_from_args(user_message, expected_output)
        else:
            raise ValueError("Either examples_file or both user_message and expected_output must be provided")
    
    def _create_llm_graph(self, model_name: Optional[str] = None) -> MetaPromptGraph:
        """Create MetaPromptGraph with specified or default model."""
        if not self.config.llms:
            raise ValueError("No LLM configurations found in config file")
        
        # Use specified model or first available
        if model_name:
            if model_name not in self.config.llms:
                raise ValueError(f"Model '{model_name}' not found in config. Available: {list(self.config.llms.keys())}")
            selected_model = model_name
        else:
            selected_model = next(iter(self.config.llms.keys()))
        
        llm_config = self.config.llms[selected_model]
        
        # Import LLM creation utilities
        from app.gradio_meta_prompt_utils import initialize_llm
        from meta_prompt.consts import META_PROMPT_NODES
        
        llm = initialize_llm(self.config, selected_model)
        llms = {node: llm for node in META_PROMPT_NODES}
        
        return MetaPromptGraph(llms=llms)
    
    def generate_prompt(self, 
                       examples: List[Example],
                       acceptance_criteria: Optional[str] = None,
                       max_output_age: Optional[int] = None,
                       recursion_limit: Optional[int] = None,
                       model_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate optimized prompt from examples."""
        
        # Create LLM graph
        self.graph = self._create_llm_graph(model_name)
        
        # Set defaults from config
        max_output_age = max_output_age or self.config.max_output_age or 2
        recursion_limit = recursion_limit or self.config.recursion_limit or 20
        
        # Generate acceptance criteria if not provided
        if not acceptance_criteria:
            acceptance_criteria = f"The response should appropriately address the user's request based on the provided examples."
        
        # Create agent state
        input_state = AgentState(
            examples=examples,
            acceptance_criteria=acceptance_criteria,
            max_output_age=max_output_age
        )
        
        # Run meta-prompt generation
        result = self.graph.run_meta_prompt_graph(input_state, recursion_limit=recursion_limit)
        
        return result
    
    def print_results(self, result: Dict[str, Any], verbose: bool = False):
        """Print results in a formatted way."""
        print("\n" + "=" * 60)
        print("🤖 META PROMPT RESULTS")
        print("=" * 60)
        
        # Status
        accepted = result.get('accepted', False)
        status_emoji = "✅" if accepted else "⚠️"
        print(f"\n{status_emoji} Status: {'Accepted' if accepted else 'Not Accepted'}")
        
        # Output age and iterations
        output_age = result.get('best_output_age', 'N/A')
        print(f"🔄 Output Age: {output_age}")
        print(f"📊 Iterations: {int(output_age) + 1 if isinstance(output_age, int) else 'N/A'}")
        
        # Generated system message
        system_message = result.get('best_system_message')
        if system_message:
            print(f"\n🤖 Generated System Message:")
            print("-" * 40)
            print(system_message)
        
        # Best output
        best_output = result.get('best_output')
        if best_output:
            print(f"\n💬 Best Generated Output:")
            print("-" * 40)
            print(best_output)
        
        # Analysis (if verbose)
        if verbose:
            analysis = result.get('analysis')
            if analysis:
                print(f"\n📈 Analysis:")
                print("-" * 40)
                try:
                    analysis_data = json.loads(analysis) if isinstance(analysis, str) else analysis
                    print(json.dumps(analysis_data, indent=2, ensure_ascii=False))
                except:
                    print(analysis)
        
        # Summary
        print(f"\n🎯 Summary:")
        print(f"   • System message generated: {'Yes' if system_message else 'No'}")
        print(f"   • Output meets criteria: {'Yes' if accepted else 'No'}")
        print(f"   • Ready for use: {'Yes' if accepted and system_message else 'No'}")
        print()


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Meta Prompt CLI - Generate optimized LLM prompts from examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate prompt from examples file
  %(prog)s examples.json

  # Generate prompt from direct input
  %(prog)s --user-message "What is the capital of France?" --expected-output "The capital of France is Paris."

  # Specify model and criteria
  %(prog)s examples.yaml --model llama3-8b-8192 --criteria "Response should be concise"
  
  # Save output to file
  %(prog)s examples.json --output result.json
  
  # Verbose output with analysis
  %(prog)s --user-message "Hello" --expected-output "Hi there!" --verbose
        """
    )
    
    # Input source arguments (mutually exclusive with direct input)
    parser.add_argument(
        'examples_file',
        nargs='?',
        help='Path to examples file (JSON or YAML format)'
    )
    
    # Direct input arguments
    parser.add_argument(
        '--user-message', '--input',
        help='User message/input for single example generation'
    )
    
    parser.add_argument(
        '--expected-output', '--output-expected',
        help='Expected output for single example generation'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config', '-c',
        default='config.yml',
        help='Configuration file path (default: config.yml)'
    )
    
    parser.add_argument(
        '--model', '-m',
        help='LLM model name to use (from config file)'
    )
    
    parser.add_argument(
        '--criteria', 
        help='Acceptance criteria for generated outputs'
    )
    
    parser.add_argument(
        '--max-output-age',
        type=int,
        help='Maximum output age for optimization iterations'
    )
    
    parser.add_argument(
        '--recursion-limit',
        type=int,
        help='Maximum recursion limit for the workflow'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file to save results (JSON format)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed analysis'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode - minimal output'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models from config and exit'
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Initialize CLI
        cli = MetaPromptCLI(config_file=args.config)
        
        # List models if requested
        if args.list_models:
            if cli.config.llms:
                print("Available models:")
                for model_name in cli.config.llms.keys():
                    print(f"  • {model_name}")
            else:
                print("No models configured in config file")
            return 0
        
        # Validate input arguments
        has_file = bool(args.examples_file)
        has_direct = bool(args.user_message and args.expected_output)
        
        if not has_file and not has_direct:
            print("Error: Either examples_file or both --user-message and --expected-output are required unless using --list-models", file=sys.stderr)
            parser.print_help()
            return 1
        
        if has_file and has_direct:
            print("Error: Cannot use both examples_file and direct input arguments simultaneously", file=sys.stderr)
            return 1
        
        if has_direct and not (args.user_message and args.expected_output):
            print("Error: Both --user-message and --expected-output are required for direct input", file=sys.stderr)
            return 1
        
        # Load examples
        if not args.quiet:
            if has_file:
                print(f"📖 Loading examples from: {args.examples_file}")
            else:
                print("📖 Using direct input example")
        
        examples = cli._load_examples(args.examples_file, args.user_message, args.expected_output)
        
        if not args.quiet:
            print(f"✅ Loaded {len(examples)} examples")
        
        # Generate prompt
        if not args.quiet:
            model_name = args.model or "default"
            print(f"🚀 Generating meta-prompt with model: {model_name}")
            print("⏳ This may take a moment...")
        
        result = cli.generate_prompt(
            examples=examples,
            acceptance_criteria=args.criteria,
            max_output_age=args.max_output_age,
            recursion_limit=args.recursion_limit,
            model_name=args.model
        )
        
        # Output results
        if not args.quiet:
            cli.print_results(result, verbose=args.verbose)
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            if not args.quiet:
                print(f"💾 Results saved to: {output_path}")
        
        # Exit with appropriate code
        return 0 if result.get('accepted', False) else 1
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())