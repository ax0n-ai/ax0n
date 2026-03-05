"""
Shared test configuration for Ax0n
"""

import sys
import os

# Add src to path so `import axon` works without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
