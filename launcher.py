#!/usr/bin/env python3
"""
GRAG Terminal Interface - Quick Start Script

This script provides a convenient way to launch the interfaces with
the correct settings for different use cases.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Menu-driven launcher."""
    
    print("\n" + "="*80)
    print("GRAG INTELLIGENT ROUTER - LAUNCHER")
    print("="*80)
    print("\nSelect an option:")
    print("\n  1. Quick Demo (recommended, no setup needed)")
    print("     - Run: python grag_router_demo.py")
    print("     - Features: Fast routing with estimated metrics")
    print("     - Setup: None required")
    print()
    print("  2. Full Production Interface (requires Ollama)")
    print("     - Run: python terminal_interface.py")
    print("     - Features: Live pipeline execution with real answers")
    print("     - Setup: Ollama running with models")
    print()
    print("  3. Show Quick Reference")
    print("     - Open: QUICK_REFERENCE.md")
    print("     - One-page cheat sheet")
    print()
    print("  4. Show Full Guide")
    print("     - Open: TERMINAL_INTERFACE_GUIDE.md")
    print("     - Complete documentation")
    print()
    print("  5. Show Index & Navigation")
    print("     - Open: INDEX.md")
    print("     - All documentation organized")
    print()
    print("  6. Exit")
    print()
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == "1":
        print("\nLaunching GRAG Router Demo...\n")
        subprocess.run([sys.executable, "grag_router_demo.py"])
    
    elif choice == "2":
        print("\nNote: Make sure Ollama is running:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull models: ollama pull nomic-embed-text qwen3:4b-instruct")
        print("\nLaunching GRAG Terminal Interface...\n")
        input("Press Enter when ready...")
        subprocess.run([sys.executable, "terminal_interface.py"])
    
    elif choice == "3":
        print("\nOpening QUICK_REFERENCE.md...")
        try:
            from pathlib import Path
            Path("QUICK_REFERENCE.md").open().read()
            print("[Note: File opened in your default viewer]")
        except:
            print("Could not open file. Please open QUICK_REFERENCE.md manually.")
    
    elif choice == "4":
        print("\nOpening TERMINAL_INTERFACE_GUIDE.md...")
        try:
            from pathlib import Path
            Path("TERMINAL_INTERFACE_GUIDE.md").open().read()
            print("[Note: File opened in your default viewer]")
        except:
            print("Could not open file. Please open TERMINAL_INTERFACE_GUIDE.md manually.")
    
    elif choice == "5":
        print("\nOpening INDEX.md...")
        try:
            from pathlib import Path
            Path("INDEX.md").open().read()
            print("[Note: File opened in your default viewer]")
        except:
            print("Could not open file. Please open INDEX.md manually.")
    
    elif choice == "6":
        print("\nGoodbye!")
    
    else:
        print("\nInvalid choice. Please run again and select 1-6.")


if __name__ == "__main__":
    main()
