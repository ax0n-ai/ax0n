"""
Basic usage example for Ax0n
Demonstrates the core functionality of the Ax0n Think & Memory Layer
"""

import asyncio
import os
from axon import Axon, AxonConfig, ReasoningMethod, LLMConfig

async def main():
    """Demonstrate basic Ax0n usage"""
    
    # Initialize Axon with configuration (matching PLAN-1.md example)
    config = AxonConfig(
        think_layer=dict(max_depth=5, enable_parallel=True, auto_iterate=True),
        llm=LLMConfig(
            provider="openai", 
            model="gpt-4", 
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key")
        ),
        memory=dict(enable_memory=False),  # Disable for demo
        grounding=dict(enable_grounding=False)  # Disable for demo
    )
    
    axon = Axon(config)
    
    # Example query
    query = "What are the benefits of using structured reasoning in AI systems?"
    
    print(f" Thinking about: {query}")
    print("=" * 60)
    
    try:
        # Generate structured thoughts using Chain of Thoughts
        result = await axon.think(query, ReasoningMethod.COT)
        
        print(f"\n Answer:\n{result['answer']}")
        print(f"\n Execution Summary:")
        print(f"  ⏱  Method: {result['method']}")
        print(f"   Thoughts: {len(result['thoughts'])} generated")
        print(f"   Complexity: {result['metadata']['method_complexity']}")
        
        if result['thoughts']:
            print("\n Reasoning Trace:")
            for i, thought in enumerate(result['thoughts'][:3], 1):  # Show first 3
                print(f"  {i}. {thought['thought'][:120]}...")
                print(f"     Stage: {thought['stage']}, Score: {thought.get('score', 'N/A')}")
        
        if result.get('citations'):
            print(f"\n Citations: {len(result['citations'])} sources found")
            for citation in result['citations'][:3]:  # Show first 3
                print(f"  - {citation.get('source_url', 'Unknown source')}")
        
    except Exception as e:
        print(f" Error: {e}")
        print("\n Make sure you have:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Installed ax0n-ai: pip install ax0n-ai")
        print("  3. Or install from source: pip install -e .")

if __name__ == "__main__":
    asyncio.run(main()) 