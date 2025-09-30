#!/bin/bash
# Virgo Neural Field Language Model - Launch Script

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the script directory to PYTHONPATH so virgo module can be imported
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

PYTHON_CMD="python3"

# Function to check if system is installed
check_installation() {
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo "Error: Python 3 is not installed or not in PATH"
        echo "Please run ./install.sh first"
        exit 1
    fi
    
    $PYTHON_CMD -c "import torch, sentence_transformers, faiss, sklearn, textblob, nltk, pytest" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Required packages are not installed"
        echo "Please run ./install.sh first"
        exit 1
    fi
}

# Function to display usage
show_usage() {
    echo "======================================"
    echo "Virgo Neural Field Launch Script"
    echo "======================================"
    echo ""
    echo "Usage: ./launch.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  chat [path]       Start interactive chat interface"
    echo "                    Optional: Specify memory storage path (default: ./memory_store)"
    echo ""
    echo "  demo              Run the demo script"
    echo ""
    echo "  evaluate          Run the evaluation script"
    echo ""
    echo "  test              Run the test suite"
    echo ""
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./launch.sh chat"
    echo "  ./launch.sh chat ./my_memory"
    echo "  ./launch.sh demo"
    echo "  ./launch.sh evaluate"
    echo "  ./launch.sh test"
    echo ""
}

# Check if command is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Parse command
COMMAND=$1
shift

case $COMMAND in
    chat)
        echo "======================================"
        echo "Starting Virgo Chat Interface"
        echo "======================================"
        echo ""
        check_installation
        
        # Get memory path if provided
        MEMORY_PATH=${1:-./memory_store}
        
        echo "Memory storage path: $MEMORY_PATH"
        echo ""
        echo "Commands:"
        echo "  Type messages to chat"
        echo "  'save'  - Train field and save memory"
        echo "  'stats' - Show memory statistics"
        echo "  'quit'  - Exit"
        echo ""
        echo "======================================"
        echo ""
        
        $PYTHON_CMD -m virgo.chat "$MEMORY_PATH"
        ;;
        
    demo)
        echo "======================================"
        echo "Running Virgo Demo"
        echo "======================================"
        echo ""
        check_installation
        
        $PYTHON_CMD scripts/demo.py
        ;;
        
    evaluate)
        echo "======================================"
        echo "Running Virgo Evaluation"
        echo "======================================"
        echo ""
        check_installation
        
        $PYTHON_CMD scripts/evaluate.py
        ;;
        
    test)
        echo "======================================"
        echo "Running Virgo Test Suite"
        echo "======================================"
        echo ""
        check_installation
        
        pytest tests/ -v
        ;;
        
    help|--help|-h)
        show_usage
        ;;
        
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        show_usage
        exit 1
        ;;
esac
