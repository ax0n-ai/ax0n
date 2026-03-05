"""
Tree of Thoughts (ToT) demonstration for Ax0n
Showcases advanced reasoning with tree-based exploration and evaluation
"""

import asyncio
import os
import json
from axon import Axon, AxonConfig, ReasoningMethod, LLMConfig

async def demonstrate_tree_of_thoughts():
    """Demonstrate Tree of Thoughts reasoning"""
    
    # Initialize Axon with Tree of Thoughts configuration
    config = AxonConfig(
        think_layer=dict(
            max_depth=5,
            enable_parallel=True,
            auto_iterate=True,
            max_branches=3,
            evaluation_threshold=0.7
        ),
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
            temperature=0.8,
            max_tokens=2000
        ),
        memory=dict(enable_memory=False),  # Disable for demo
        grounding=dict(enable_grounding=False)  # Disable for demo
    )
    
    axon = Axon(config)
    
    # Complex problem that benefits from multiple reasoning paths
    query = """
    A company needs to decide whether to:
    1. Expand into a new market (requires $2M investment, 50% success rate)
    2. Improve existing products (requires $500K investment, 80% success rate)
    3. Acquire a competitor (requires $5M investment, 30% success rate)
    
    The company has $3M available and wants to maximize ROI while minimizing risk.
    What should they do and why?
    """
    
    print(" Tree of Thoughts Reasoning Demo")
    print("=" * 60)
    print(f" Problem: {query.strip()}")
    print("=" * 60)
    
    try:
        # Generate thoughts using Tree of Thoughts
        result = await axon.think(query, ReasoningMethod.TOT)
        
        print(f"\n Final Answer:")
        print(f"{result['answer']}")
        
        print(f"\n Execution Summary:")
        print(f"  ⏱  Method: {result['method']}")
        print(f"   Thoughts: {len(result['thoughts'])}")
        branches = set(t.get('branch_id') for t in result['thoughts'] if t.get('branch_id'))
        print(f"   Branches explored: {len(branches)}")
        
        # Display the reasoning tree
        print(f"\n Reasoning Tree:")
        print_tree_structure(result['thoughts'])
        
        # Show thought evaluations
        print(f"\n Thought Evaluations:")
        for i, thought in enumerate(result['thoughts'], 1):
            evaluation = thought.get('metadata', {}).get('evaluation', 'unknown')
            score = thought.get('score', 0.0)
            print(f"  {i}. {thought['thought'][:80]}...")
            print(f"     Evaluation: {evaluation}, Score: {score:.2f}")
            if thought.get('branch_id'):
                print(f"     Branch: {thought['branch_id']}")
            print()
        
        # Compare with sequential reasoning
        print(f"\n Comparison: Sequential vs Tree of Thoughts")
        print("-" * 40)
        
        # Sequential reasoning (Chain of Thoughts)
        seq_result = await axon.think(query, ReasoningMethod.COT)
        
        print(f"Sequential Reasoning (CoT):")
        print(f"  Thoughts: {len(seq_result['thoughts'])}")
        print(f"  Method: {seq_result['method']}")
        print(f"  Answer: {seq_result['answer'][:100]}...")
        
        print(f"\nTree of Thoughts:")
        print(f"  Thoughts: {len(result['thoughts'])}")
        print(f"  Method: {result['method']}")
        print(f"  Answer: {result['answer'][:100]}...")
        
        # Show the difference in approach
        print(f"\n Key Differences:")
        print(f"  • ToT explores multiple reasoning paths simultaneously")
        print(f"  • ToT evaluates and prunes branches based on quality")
        print(f"  • ToT can backtrack and revise earlier thoughts")
        print(f"  • ToT provides more comprehensive analysis")
        
    except Exception as e:
        print(f" Error: {e}")
        print("\n Make sure you have:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Installed ax0n-ai: pip install ax0n-ai")
        print("  3. Or install from source: pip install -e .")


def print_tree_structure(thoughts: list):
    """Print the tree structure of thoughts"""
    if not thoughts:
        return
    
    # Group thoughts by branch
    branches = {}
    for thought in thoughts:
        branch_id = thought.get('branch_id', 'main')
        if branch_id not in branches:
            branches[branch_id] = []
        branches[branch_id].append(thought)
    
    # Print each branch
    for branch_id, branch_thoughts in branches.items():
        print(f"\n Branch: {branch_id}")
        for i, thought in enumerate(branch_thoughts, 1):
            prefix = "  " * (i - 1) + "├─ " if i < len(branch_thoughts) else "  " * (i - 1) + "└─ "
            thought_num = thought.get('thought_number', i)
            thought_text = thought.get('thought', 'No thought')
            print(f"{prefix}Step {thought_num}: {thought_text[:60]}...")


async def demonstrate_branching_scenarios():
    """Demonstrate different branching scenarios"""
    
    config = AxonConfig(
        think_layer=dict(max_depth=3, enable_parallel=True, max_branches=3),
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key")
        ),
        memory=dict(enable_memory=False),
        grounding=dict(enable_grounding=False)
    )
    
    axon = Axon(config)
    
    scenarios = [
        {
            "name": "Decision Making",
            "query": "Should a startup pivot to a new market or double down on their current product?"
        },
        {
            "name": "Problem Solving", 
            "query": "How can we reduce carbon emissions in urban transportation while maintaining economic growth?"
        },
        {
            "name": "Creative Writing",
            "query": "Write a story about a time traveler who discovers they can't return to their original timeline."
        }
    ]
    
    print("\n Branching Scenarios Demo")
    print("=" * 60)
    
    for scenario in scenarios:
        print(f"\n Scenario: {scenario['name']}")
        print(f"Query: {scenario['query']}")
        
        try:
            result = await axon.think(scenario['query'], ReasoningMethod.TOT)
            
            print(f" Answer: {result['answer'][:150]}...")
            branches = set(t.get('branch_id') for t in result['thoughts'] if t.get('branch_id'))
            print(f" Branches: {len(branches)}")
            print(f"⏱  Method: {result['method']}")
            
        except Exception as e:
            print(f" Error: {e}")


if __name__ == "__main__":
    print(" Ax0n Tree of Thoughts Demo")
    print("This demo shows how Tree of Thoughts enables advanced reasoning")
    print("by exploring multiple paths simultaneously and evaluating each branch.")
    
    asyncio.run(demonstrate_tree_of_thoughts())
    asyncio.run(demonstrate_branching_scenarios()) 