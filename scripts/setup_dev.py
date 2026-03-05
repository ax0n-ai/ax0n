#!/usr/bin/env python3
"""
Development environment setup script for Ax0n
Sets up virtual environment, installs dependencies, and configures development tools
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and check:
            print(result.stderr)
        print(f" {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("\n Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f" Python 3.8+ required, but found {version.major}.{version.minor}")
        return False
    
    print(f" Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment():
    """Create a virtual environment"""
    venv_path = Path('.venv')
    
    if venv_path.exists():
        print("\n Virtual environment already exists at .venv")
        print("   Remove it and recreate? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_path)
            print("   Removed existing virtual environment")
        else:
            return True
    
    return run_command(
        [sys.executable, '-m', 'venv', '.venv'],
        "Creating virtual environment"
    )

def get_venv_python():
    """Get the path to the virtual environment's Python"""
    if platform.system() == 'Windows':
        return Path('.venv') / 'Scripts' / 'python.exe'
    else:
        return Path('.venv') / 'bin' / 'python'

def install_package_editable(venv_python):
    """Install the package in editable mode"""
    return run_command(
        [str(venv_python), '-m', 'pip', 'install', '-e', '.'],
        "Installing package in editable mode"
    )

def install_dev_dependencies(venv_python):
    """Install development dependencies"""
    return run_command(
        [str(venv_python), '-m', 'pip', 'install', '-e', '.[dev,docs]'],
        "Installing development dependencies"
    )

def setup_pre_commit(venv_python):
    """Set up pre-commit hooks"""
    print("\n Setting up pre-commit hooks...")
    
    if not run_command(
        [str(venv_python), '-m', 'pip', 'install', 'pre-commit'],
        "Installing pre-commit",
        check=False
    ):
        return False
    
    # Check if .pre-commit-config.yaml exists
    if not Path('.pre-commit-config.yaml').exists():
        print("     .pre-commit-config.yaml not found, skipping hook installation")
        return True
    
    return run_command(
        [str(venv_python), '-m', 'pre_commit', 'install'],
        "Installing pre-commit hooks"
    )

def run_initial_tests(venv_python):
    """Run initial tests to verify setup"""
    print("\n Running initial tests...")
    
    return run_command(
        [str(venv_python), '-m', 'pytest', 'tests/', '-v', '--tb=short'],
        "Running test suite",
        check=False
    )

def create_env_template():
    """Create .env.template file"""
    env_template = Path('.env.template')
    
    if env_template.exists():
        print("\n .env.template already exists")
        return
    
    print("\n Creating .env.template...")
    
    content = """# Ax0n Environment Variables Template
# Copy this to .env and fill in your values

# LLM API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Vector Database Configuration
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-weaviate-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=your-pinecone-environment-here

# Search Provider Configuration
GOOGLE_API_KEY=your-google-api-key-here
GOOGLE_CSE_ID=your-google-cse-id-here

# Optional: Custom Configuration
AX0N_LOG_LEVEL=INFO
AX0N_DEBUG=false
"""
    
    env_template.write_text(content)
    print(" Created .env.template")
    print("   Copy this to .env and fill in your API keys")

def print_activation_instructions():
    """Print instructions for activating the virtual environment"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║    Development Environment Setup Complete!             ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("\n Next Steps:")
    print("\n1. Activate the virtual environment:")
    
    if platform.system() == 'Windows':
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    
    print("\n2. Copy .env.template to .env and fill in your API keys:")
    print("   cp .env.template .env")
    
    print("\n3. Run tests to verify everything works:")
    print("   pytest tests/")
    
    print("\n4. Start developing!")
    print("   python examples/basic_usage.py")
    
    print("\n Useful Commands:")
    print("   pytest tests/                    # Run all tests")
    print("   pytest tests/ -v                 # Run tests with verbose output")
    print("   black src/ tests/                # Format code")
    print("   flake8 src/ tests/               # Lint code")
    print("   mypy src/                        # Type check")
    print("   python scripts/build_and_publish.py build   # Build package")

def main():
    """Main setup workflow"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   Ax0n Development Environment Setup                     ║
    ║   Model-Agnostic Think & Memory Layer for LLMs          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check if we're in the right directory
    if not Path('pyproject.toml').exists():
        print(" Error: pyproject.toml not found. Are you in the project root?")
        sys.exit(1)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("\n Failed to create virtual environment")
        sys.exit(1)
    
    venv_python = get_venv_python()
    
    # Upgrade pip
    run_command(
        [str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip'],
        "Upgrading pip"
    )
    
    # Install package in editable mode
    if not install_package_editable(venv_python):
        print("\n Failed to install package")
        sys.exit(1)
    
    # Install development dependencies
    if not install_dev_dependencies(venv_python):
        print("\n  Failed to install some development dependencies")
    
    # Setup pre-commit hooks
    setup_pre_commit(venv_python)
    
    # Create .env template
    create_env_template()
    
    # Run initial tests
    print("\n" + "="*60)
    print("Would you like to run initial tests? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        run_initial_tests(venv_python)
    
    # Print activation instructions
    print_activation_instructions()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Setup cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
