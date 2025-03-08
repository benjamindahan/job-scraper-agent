"""
Test script for CV file upload functionality.
"""

import asyncio
import sys
import os
import tempfile

# Add parent directory to path to import modules
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.file_utils import extract_text_from_file
from src.cv_upload import process_cv_file, process_cv_file_and_invoke_graph
from src.graph import graph, initial_state_creator

# Test CV content (as a string)
TEST_CV = """
JANE DOE
Data Scientist with 5 years of experience

SKILLS
Python, SQL, Machine Learning, Deep Learning, NLP, Data Visualization, PyTorch, TensorFlow

EXPERIENCE
Senior Data Scientist at AI Solutions Inc. (2022-Present)
- Led development of NLP models for sentiment analysis
- Improved model accuracy by 25% using transformer architectures

Data Scientist at Tech Corp (2020-2022)
- Built predictive models for customer churn
- Analyzed user behavior using SQL and Python

EDUCATION
M.S. Computer Science, Stanford University (2020)
B.S. Mathematics, MIT (2018)
"""


async def test_file_extraction():
    """Test the file extraction utility."""
    print("=== Testing File Extraction ===")

    # Create a temporary text file with our CV content
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False,
                                     mode='w') as f:
        f.write(TEST_CV)
        temp_file_path = f.name

    try:
        print(f"Created temporary CV file: {temp_file_path}")

        # Test extraction
        result = extract_text_from_file(file_path=temp_file_path)

        if result['success']:
            print("✅ Text extraction successful")
            print(f"Extracted {len(result['text'])} characters")
            print("\nPreview of extracted text:")
            print(result['text'][:200] + "...")
        else:
            print(f"❌ Text extraction failed: {result['error']}")

        return result['success']

    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"Deleted temporary file: {temp_file_path}")


async def test_process_cv_file():
    """Test the process_cv_file function."""
    print("\n=== Testing CV File Processing ===")

    # Create a temporary text file with our CV content
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False,
                                     mode='w') as f:
        f.write(TEST_CV)
        temp_file_path = f.name

    try:
        print(f"Created temporary CV file: {temp_file_path}")

        # Initialize a state
        state = initial_state_creator()
        state["job_title"] = "data scientist"
        state["max_jobs"] = 5

        # Process the CV file
        result_state = await process_cv_file(
            file_path=temp_file_path,
            current_state=state
        )

        if 'error' in result_state:
            print(f"❌ CV file processing failed: {result_state['error']}")
            return False

        print("✅ CV file processing successful")
        print(
            f"State now contains user_input with {len(result_state['user_input'])} characters")
        print(f"CV file path stored: {result_state.get('cv_file_path')}")

        return True

    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"Deleted temporary file: {temp_file_path}")


async def test_integration():
    """
    Test integration with the graph (just the state preparation).

    This test doesn't actually invoke the graph to avoid lengthy scraping,
    but it verifies that we can prepare the state correctly.
    """
    print("\n=== Testing Integration ===")

    # Create a temporary text file with our CV content
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False,
                                     mode='w') as f:
        f.write(TEST_CV)
        temp_file_path = f.name

    try:
        print(f"Created temporary CV file: {temp_file_path}")

        # Initialize a state that would normally come from earlier steps
        state = initial_state_creator()
        state["job_title"] = "data scientist"
        state["max_jobs"] = 5
        state["waiting_for_cv"] = True  # This is normally set by the graph

        # Process the CV file (but don't invoke the graph)
        result_state = await process_cv_file(
            file_path=temp_file_path,
            current_state=state
        )

        if 'error' in result_state:
            print(f"❌ Integration test failed: {result_state['error']}")
            return False

        print("✅ Integration test successful")
        print("State is prepared correctly for graph invocation")

        # Verify that the modified collect_cv function would be able to process this state
        if result_state.get('user_input') and result_state.get('cv_file_path'):
            print(
                "✅ State contains both user_input and cv_file_path, ready for collect_cv")
        else:
            print("❌ State missing required fields")
            return False

        return True

    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"Deleted temporary file: {temp_file_path}")


async def run_tests():
    """Run all tests."""
    tests = [
        ("File Extraction", test_file_extraction),
        ("CV File Processing", test_process_cv_file),
        ("Integration", test_integration)
    ]

    results = []

    for name, test_func in tests:
        print(f"\n{'=' * 50}")
        print(f"Running test: {name}")
        print(f"{'=' * 50}")

        success = await test_func()
        results.append((name, success))

    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)

    all_passed = True
    for name, success in results:
        result = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {result}")
        all_passed = all_passed and success

    return all_passed


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)