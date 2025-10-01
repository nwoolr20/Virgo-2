#!/usr/bin/env python3
"""
Virgo-2 System Test Script

This script runs a comprehensive system test following the QUICKSTART guide:
1. Install the package using pip install -e .
2. Download required data (NLTK, sentence-transformers model)
3. Verify installation with test suite
4. Run the interactive demo
5. Run comprehensive evaluation
6. Try the chat interface (automated)

All results are saved to "V2 System Test.md"
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import tempfile
import time


class SystemTestRunner:
    def __init__(self):
        self.output_file = Path("V2 System Test.md")
        self.output_lines = []
        self.test_start_time = datetime.now()
        
    def log(self, message, level="INFO"):
        """Log a message to console and output list."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)
        self.output_lines.append(f"{message}\n")
        
    def log_section(self, title):
        """Log a section header."""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80 + "\n")
        self.output_lines.append(f"\n## {title}\n\n")
        
    def run_command(self, command, description, timeout=300, capture_output=True):
        """Run a command and capture its output."""
        self.log(f"Running: {description}")
        self.output_lines.append(f"**Command:** `{command}`\n\n")
        
        try:
            start_time = time.time()
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd="/home/runner/work/Virgo-2/Virgo-2"
            )
            elapsed_time = time.time() - start_time
            
            if result.stdout:
                self.output_lines.append(f"**Output:**\n```\n{result.stdout}\n```\n\n")
                print(result.stdout)
                
            if result.stderr:
                self.output_lines.append(f"**Errors/Warnings:**\n```\n{result.stderr}\n```\n\n")
                print(result.stderr, file=sys.stderr)
            
            self.output_lines.append(f"**Elapsed Time:** {elapsed_time:.2f} seconds\n")
            self.output_lines.append(f"**Exit Code:** {result.returncode}\n")
            
            if result.returncode == 0:
                self.log(f"✓ {description} - SUCCESS", "SUCCESS")
                self.output_lines.append(f"**Status:** ✓ SUCCESS\n\n")
            else:
                self.log(f"✗ {description} - FAILED (exit code {result.returncode})", "ERROR")
                self.output_lines.append(f"**Status:** ✗ FAILED\n\n")
                
            return result.returncode == 0, result
            
        except subprocess.TimeoutExpired:
            self.log(f"✗ {description} - TIMEOUT (>{timeout}s)", "ERROR")
            self.output_lines.append(f"**Status:** ✗ TIMEOUT (>{timeout}s)\n\n")
            return False, None
        except Exception as e:
            self.log(f"✗ {description} - ERROR: {e}", "ERROR")
            self.output_lines.append(f"**Status:** ✗ ERROR: {e}\n\n")
            return False, None
            
    def step1_install_package(self):
        """Step 1: Install the package using pip install -e ."""
        self.log_section("Step 1: Install Package")
        
        # First, ensure pip, setuptools, and wheel are up to date
        self.log("Upgrading pip, setuptools, and wheel...")
        self.run_command(
            "pip install --upgrade pip setuptools wheel",
            "Upgrade pip, setuptools, and wheel",
            timeout=120
        )
        
        # Install dependencies first
        self.log("Installing dependencies...")
        success_deps, _ = self.run_command(
            "pip install torch sentence-transformers faiss-cpu scikit-learn textblob pytest nltk",
            "Install Python dependencies",
            timeout=300
        )
        
        # Then install package in editable mode
        return self.run_command(
            "pip install -e .",
            "Install package in editable mode",
            timeout=300
        )
        
    def step2_download_data(self):
        """Step 2: Download required data (NLTK, sentence-transformers model)."""
        self.log_section("Step 2: Download Required Data")
        
        # Download NLTK data
        success1, _ = self.run_command(
            'python -c "import nltk; nltk.download(\'brown\', quiet=True); nltk.download(\'punkt\', quiet=True); print(\'✓ NLTK data downloaded\')"',
            "Download NLTK data",
            timeout=120
        )
        
        # Download sentence-transformers model
        success2, _ = self.run_command(
            'python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer(\'all-MiniLM-L6-v2\'); print(\'✓ Sentence-transformers model loaded\')"',
            "Download sentence-transformers model",
            timeout=300
        )
        
        return success1 and success2
        
    def step3_verify_installation(self):
        """Step 3: Verify installation with test suite."""
        self.log_section("Step 3: Verify Installation with Test Suite")
        return self.run_command(
            "pytest tests/ -v",
            "Run test suite",
            timeout=300
        )
        
    def step4_run_demo(self):
        """Step 4: Run the interactive demo."""
        self.log_section("Step 4: Run Interactive Demo")
        return self.run_command(
            "python scripts/demo.py",
            "Run demo script",
            timeout=300
        )
        
    def step5_run_evaluation(self):
        """Step 5: Run comprehensive evaluation."""
        self.log_section("Step 5: Run Comprehensive Evaluation")
        return self.run_command(
            "python scripts/evaluate.py",
            "Run evaluation script",
            timeout=300
        )
        
    def step6_try_chat_interface(self):
        """Step 6: Try the chat interface (automated test)."""
        self.log_section("Step 6: Try Chat Interface (Automated Test)")
        
        # Create a test script that simulates chat interaction
        test_chat_script = '''
import sys
import tempfile
from pathlib import Path
from virgo import MemorySystem

# Create temporary memory store
temp_dir = Path(tempfile.mkdtemp())
print(f"Using temporary memory store: {temp_dir}")

# Initialize system
print("\\n=== Initializing Memory System ===")
system = MemorySystem()

# Add some test memories
print("\\n=== Adding Test Memories ===")
test_memories = [
    "My name is Alice",
    "I love programming in Python",
    "I work as a software engineer",
    "My favorite color is blue",
    "I have a cat named Whiskers"
]

for i, memory in enumerate(test_memories, 1):
    system.store(memory, speaker_id=0)
    print(f"  {i}. Stored: {memory}")

# Train the field
print("\\n=== Training Neural Field ===")
system.fit_field(num_steps=1000, verbose=False)
print("✓ Field trained successfully")

# Test retrieval
print("\\n=== Testing Retrieval ===")
test_queries = [
    "What is my name?",
    "What programming language do I like?",
    "Tell me about my pet"
]

for query in test_queries:
    print(f"\\nQuery: {query}")
    results = system.retrieve(query, k=2)
    print("Results:")
    for i, (memory, distance) in enumerate(results, 1):
        print(f"  {i}. [distance={distance:.2f}] {memory.text}")

# Show stats
print("\\n=== System Statistics ===")
stats = system.get_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")

# Save system
print("\\n=== Saving System ===")
system.save(temp_dir)
print(f"✓ System saved to {temp_dir}")

# Load system
print("\\n=== Loading System ===")
system2 = MemorySystem()
system2.load(temp_dir)
print("✓ System loaded successfully")

# Verify loaded system
print("\\n=== Verifying Loaded System ===")
stats2 = system2.get_stats()
print(f"Loaded memories: {stats2['total_memories']}")
print(f"Field fitted: {stats2.get('is_fitted', 'N/A')}")

# Clean up
import shutil
shutil.rmtree(temp_dir)
print(f"\\n✓ Cleaned up temporary directory")
print("\\n=== Chat Interface Test Complete ===")
'''
        
        # Write test script to temp file
        temp_script = Path("/tmp/test_chat_interface.py")
        temp_script.write_text(test_chat_script)
        
        return self.run_command(
            f"python {temp_script}",
            "Automated chat interface test",
            timeout=300
        )
        
    def generate_summary(self, results):
        """Generate test summary."""
        self.log_section("Test Summary")
        
        total_tests = len(results)
        passed = sum(1 for r in results if r[1])
        failed = total_tests - passed
        
        summary = f"""
**Total Steps:** {total_tests}
**Passed:** {passed} ✓
**Failed:** {failed} ✗
**Success Rate:** {(passed/total_tests)*100:.1f}%

### Detailed Results:

"""
        self.output_lines.append(summary)
        
        for step_name, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            self.output_lines.append(f"- {step_name}: {status}\n")
            
        # Overall status
        self.output_lines.append(f"\n### Overall Status\n\n")
        if failed == 0:
            self.output_lines.append(f"**✓ ALL TESTS PASSED**\n\n")
            self.log("✓ ALL TESTS PASSED", "SUCCESS")
        else:
            self.output_lines.append(f"**✗ SOME TESTS FAILED ({failed}/{total_tests})**\n\n")
            self.log(f"✗ SOME TESTS FAILED ({failed}/{total_tests})", "WARNING")
            
        # Test duration
        total_duration = (datetime.now() - self.test_start_time).total_seconds()
        self.output_lines.append(f"**Total Test Duration:** {total_duration:.2f} seconds\n\n")
        
    def save_results(self):
        """Save all results to markdown file."""
        self.log_section("Saving Results")
        
        # Create the full markdown document
        markdown_content = f"""# V2 System Test

**Test Date:** {self.test_start_time.strftime("%Y-%m-%d %H:%M:%S")}
**Test Duration:** {(datetime.now() - self.test_start_time).total_seconds():.2f} seconds

This document contains the complete results of the Virgo-2 system test,
following the steps outlined in the QUICKSTART guide.

---

"""
        markdown_content += "".join(self.output_lines)
        
        # Write to file
        self.output_file.write_text(markdown_content)
        self.log(f"✓ Results saved to {self.output_file.absolute()}")
        
    def run(self):
        """Run all test steps."""
        print("\n" + "="*80)
        print("  VIRGO-2 COMPREHENSIVE SYSTEM TEST")
        print("="*80 + "\n")
        
        results = []
        
        # Step 1: Install package
        success, _ = self.step1_install_package()
        results.append(("Step 1: Install Package", success))
        if not success:
            self.log("Installation failed, continuing anyway...", "WARNING")
        
        # Step 2: Download data
        success = self.step2_download_data()
        results.append(("Step 2: Download Required Data", success))
        if not success:
            self.log("Data download had issues, continuing anyway...", "WARNING")
        
        # Step 3: Verify installation
        success, _ = self.step3_verify_installation()
        results.append(("Step 3: Verify Installation", success))
        
        # Step 4: Run demo
        success, _ = self.step4_run_demo()
        results.append(("Step 4: Run Demo", success))
        
        # Step 5: Run evaluation
        success, _ = self.step5_run_evaluation()
        results.append(("Step 5: Run Evaluation", success))
        
        # Step 6: Try chat interface
        success, _ = self.step6_try_chat_interface()
        results.append(("Step 6: Chat Interface Test", success))
        
        # Generate summary
        self.generate_summary(results)
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("  SYSTEM TEST COMPLETE")
        print("="*80 + "\n")


if __name__ == "__main__":
    runner = SystemTestRunner()
    runner.run()
