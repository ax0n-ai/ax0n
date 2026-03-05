"""
Multiple Reasoning Methods demonstration for Ax0n
Showcases all available reasoning methods with the same query
"""

import asyncio
import os
from axon import Axon, AxonConfig, ReasoningMethod, LLMConfig

async def main():
    """Demonstrate all reasoning methods with a single query"""
    
    # Initialize Axon with full configuration
    config = AxonConfig(
        think_layer=dict(
            max_depth=4,
            enable_parallel=True,
            auto_iterate=True,
            max_branches=3,
            evaluation_threshold=0.7
        ),
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
            temperature=0.7,
            max_tokens=1500
        ),
        memory=dict(enable_memory=False),  # Disable for demo
        grounding=dict(enable_grounding=False)  # Disable for demo
    )
    
    axon = Axon(config)
    
    # Example query that benefits from different reasoning approaches
    query = "How can a small tech startup effectively compete with established companies in their market?"
    
    print(" Ax0n Multiple Reasoning Methods Demo")
    print("=" * 70)
    print(f" Query: {query}")
    print("=" * 70)
    
    # List of all reasoning methods to demonstrate
    methods = [
        (ReasoningMethod.COT, "Chain of Thoughts", "Sequential step-by-step reasoning"),
        (ReasoningMethod.SELF_CONSISTENCY, "Self-Consistency", "Multiple parallel paths with voting"),
        (ReasoningMethod.AOT, "Algorithm of Thoughts", "Algorithmic problem decomposition"),
        (ReasoningMethod.TOT, "Tree of Thoughts", "Tree-based exploration with evaluation"),
    ]
    
    results = {}
    
    for method, name, description in methods:
        print(f"\n{'='*70}")
        print(f" Method: {name}")
        print(f" Description: {description}")
        print(f"{'='*70}")
        
        try:
            result = await axon.think(query, method)
            results[name] = result
            
            print(f"\n Answer:")
            print(f"{result['answer'][:300]}...")
            
            print(f"\n Statistics:")
            print(f"  • Method: {result['method']}")
            print(f"  • Thoughts Generated: {len(result['thoughts'])}")
            print(f"  • Complexity: {result['metadata']['method_complexity']}")
            
            # Show first 2 thoughts
            if result['thoughts']:
                print(f"\n Sample Thoughts:")
                for i, thought in enumerate(result['thoughts'][:2], 1):
                    print(f"  {i}. {thought['thought'][:100]}...")
                    print(f"     Stage: {thought['stage']}")
            
        except Exception as e:
            print(f" Error with {name}: {e}")
    
    # Comparative Analysis
    print(f"\n{'='*70}")
    print(" Comparative Analysis")
    print(f"{'='*70}")
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Thoughts: {len(result['thoughts'])}")
        print(f"  Answer Length: {len(result['answer'])} chars")
        print(f"  Complexity: {result['metadata']['method_complexity']}")
    
    # Method Recommendations
    print(f"\n{'='*70}")
    print(" Method Recommendations")
    print(f"{'='*70}")
    print("\nUse Chain of Thoughts (CoT) when:")
    print("  • You need simple, sequential reasoning")
    print("  • Speed is more important than comprehensive analysis")
    print("  • The problem is straightforward")
    
    print("\nUse Self-Consistency when:")
    print("  • You need high confidence in the answer")
    print("  • Multiple perspectives would be valuable")
    print("  • Consensus validation is important")
    
    print("\nUse Algorithm of Thoughts (AoT) when:")
    print("  • The problem has clear algorithmic steps")
    print("  • You're solving math or optimization problems")
    print("  • Structured decomposition is beneficial")
    
    print("\nUse Tree of Thoughts (ToT) when:")
    print("  • You need to explore multiple reasoning paths")
    print("  • The problem benefits from evaluation and backtracking")
    print("  • Planning or complex problem-solving is required")
    
    print(f"\n{'='*70}")
    print(" Demo Complete!")
    print(f"{'='*70}")
    
    # Get method comparison from Axon
    print("\n Available Methods from Axon:")
    methods_info = axon.get_available_methods()
    for method_info in methods_info:
        print(f"  • {method_info['name']}: {method_info['description']}")
        print(f"    Complexity: {method_info['complexity']}, Calls: {method_info['calls']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n Demo interrupted by user")
    except Exception as e:
        print(f"\n Demo failed: {e}")
        print("\n Make sure you have:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Installed ax0n-ai: pip install ax0n-ai")
        print("  3. Or install from source: pip install -e .")

