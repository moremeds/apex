#!/usr/bin/env python
"""
Setup script for Apex development environment using uv.
This script initializes the development environment and verifies dependencies.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        result = run_command(["uv", "--version"], check=False)
        if result.returncode == 0:
            print(f"✓ uv is installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    return False


def install_uv():
    """Install uv if not present."""
    print("Installing uv...")
    if sys.platform == "win32":
        print("Please install uv using: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        return False
    
    result = run_command(
        ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
        check=False
    )
    if result.returncode != 0:
        print(f"Failed to install uv: {result.stderr}")
        return False
    return True


def setup_virtual_environment():
    """Create and setup virtual environment using uv."""
    print("\n📦 Setting up virtual environment...")
    
    # Create venv if it doesn't exist
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        run_command(["uv", "venv"])
        print("✓ Virtual environment created")
    else:
        print("✓ Virtual environment already exists")


def install_dependencies():
    """Install project dependencies using uv."""
    print("\n📦 Installing dependencies...")
    
    # Sync with lock file if it exists
    if Path("uv.lock").exists():
        print("Syncing with uv.lock...")
        run_command(["uv", "sync"])
        print("✓ Dependencies synced")
    else:
        print("Installing from pyproject.toml...")
        run_command(["uv", "pip", "install", "-e", ".[dev,ml]"])
        print("✓ Dependencies installed")


def create_directories():
    """Create necessary directories for the project."""
    print("\n📁 Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/cache",
        "results/backtests",
        "results/reports",
        "logs",
        ".cache"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")


def verify_installation():
    """Verify that the installation is working."""
    print("\n🔍 Verifying installation...")
    
    # Check if we can import the package
    result = run_command(
        ["uv", "run", "python", "-c", "import apex; print('✓ Apex package importable')"],
        check=False
    )
    
    if result.returncode != 0:
        print("⚠️  Warning: Could not import apex package")
        print(result.stderr)
    else:
        print(result.stdout.strip())
    
    # Check key dependencies
    deps_to_check = ["polars", "vectorbt", "pandas", "numpy"]
    for dep in deps_to_check:
        result = run_command(
            ["uv", "run", "python", "-c", f"import {dep}; print('✓ {dep} installed')"],
            check=False
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"⚠️  Warning: {dep} not installed")


def main():
    """Main setup function."""
    print("🚀 Setting up Apex development environment")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 12):
        print(f"❌ Python 3.12+ required, you have {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Check and install uv
    if not check_uv_installed():
        print("uv not found, installing...")
        if not install_uv():
            print("❌ Failed to install uv. Please install manually.")
            sys.exit(1)
    
    # Setup environment
    setup_virtual_environment()
    install_dependencies()
    create_directories()
    verify_installation()
    
    print("\n" + "=" * 50)
    print("✅ Environment setup complete!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    print("   source .venv/bin/activate  # Unix/macOS")
    print("   .venv\\Scripts\\activate     # Windows")
    print("2. Run tests: make test")
    print("3. Start developing!")


if __name__ == "__main__":
    main()