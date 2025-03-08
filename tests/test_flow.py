"""
Test script for the end-to-end flow of the Job Scraper Agent.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph import graph, initial_state_creator


def log_test(msg: str):
    print(f"[TEST LOG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


async def test_flow():
    log_test("=== Testing Job Scraper Agent Flow ===")

    # Step 1: Initial state and job scraping
    log_test("Step 1: Scraping jobs")
    state = initial_state_creator()
    state["job_title"] = "data scientist"
    state["max_jobs"] = 5  # Limit to 5 for testing

    config = {"configurable": {"thread_id": "test-flow-thread"}}
    result = await graph.ainvoke(state, config=config)
    log_test(f"Found {len(result.get('jobs_data', []))} jobs")

    # Step 2: Set user preferences
    log_test("Step 2: Setting user preferences")
    result[
        "user_input"] = "2 years experience in Tel Aviv, posted in the last month"
    result["waiting_for_preferences"] = True

    result = await graph.ainvoke(result, config=config)
    log_test(
        f"Filtered to {len(result.get('filtered_jobs', []))} jobs based on preferences")

    # Step 3: Process CV
    log_test("Step 3: Processing CV")
    sample_cv = """
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

    result["user_input"] = sample_cv
    result["waiting_for_cv"] = True
    result["waiting_for_preferences"] = False  # Don't reprocess preferences

    result = await graph.ainvoke(result, config=config)
    log_test(f"Ranked {len(result.get('ranked_jobs', []))} jobs based on CV")

    # Step 4: Select jobs
    log_test("Step 4: Selecting jobs")
    result["user_input"] = "1"  # Select first job
    result["waiting_for_job_selection"] = True
    result["waiting_for_cv"] = False  # Don't reprocess CV

    result = await graph.ainvoke(result, config=config)
    log_test(
        f"Selected {len(result.get('selected_jobs', []))} jobs for optimization")

    # Check final results
    log_test("Test completed. Checking final state...")
    if result.get("optimized_cvs"):
        log_test(
            f"Successfully optimized {len(result.get('optimized_cvs'))} CVs")
        return True
    else:
        log_test("No optimized CVs were created")
        return False


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    success = asyncio.run(test_flow())
    sys.exit(0 if success else 1)