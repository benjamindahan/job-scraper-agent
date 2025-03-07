"""
Automated tests for the Job Application Assistant.
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph import graph, initial_state_creator

def log_test(msg: str):
    print(f"[TEST LOG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

async def run_automated_test():
    log_test("=== Automated Test: Job Application Assistant STARTED ===")
    # Initial state
    state = initial_state_creator()
    state["job_title"] = "data scientist"
    state["max_jobs"] = 10
    log_test(f"Initial state set: job_title={state['job_title']}, max_jobs={state['max_jobs']}")

    log_test(f"1. Searching for '{state['job_title']}' jobs (max: {state['max_jobs']})...")
    config = {"configurable": {"thread_id": "test-thread"}}

    try:
        # Start the graph
        result = await graph.ainvoke(state, config=config)
        log_test("Graph invoked first time. Result keys: " + ", ".join(result.keys()))
        if "state" not in result:
            log_test(
                "Warning: 'state' key not found in result. (Full result suppressed)")
        current_state = result.get("state", result)
        log_test("Current state after first invoke: " + ", ".join(current_state.keys()))

        # Process preferences if needed
        if current_state.get("waiting_for_preferences"):
            log_test("2. Setting job preferences automatically...")
            current_state["user_input"] = "I'm looking for jobs with at least 3 years of experience, posted in the Last Month, in Tel Aviv"
            result = await graph.ainvoke(current_state, config=config)
            log_test("Graph invoked after preferences. Result keys: " + ", ".join(result.keys()))
            current_state = result.get("state", result)
            log_test("Current state after preferences: " + ", ".join(current_state.keys()))

        # Process CV input if needed
        if current_state.get("waiting_for_cv"):
            log_test("3. Providing sample CV...")
            current_state["user_input"] = """
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
            result = await graph.ainvoke(current_state, config=config)
            log_test("Graph invoked after CV input. Result keys: " + ", ".join(result.keys()))
            current_state = result.get("state", result)
            log_test("Current state after CV input: " + ", ".join(current_state.keys()))

        # Process job selection if needed
        if current_state.get("waiting_for_job_selection"):
            log_test("4. Selecting top jobs automatically...")
            current_state["user_input"] = "1, 2"
            result = await graph.ainvoke(current_state, config=config)
            log_test("Graph invoked after job selection. Result keys: " + ", ".join(result.keys()))
            current_state = result.get("state", result)
            log_test("Current state after job selection: " + ", ".join(current_state.keys()))

        log_test("Automated test completed. Final state keys: " + ", ".join(current_state.keys()))
        print_results(current_state)
        return current_state
    except Exception as e:
        log_test("Error during test: " + str(e))
        traceback.print_exc()
        return {"error": str(e)}

def print_results(state):
    print("\n=== Test Results ===")
    if state.get("error"):
        print(f"Error encountered: {state['error']}")
        return
    if state.get("jobs_data"):
        print(f"Found and processed {len(state['jobs_data'])} jobs")
    if state.get("filtered_jobs"):
        print(f"Filtered to {len(state['filtered_jobs'])} matching jobs")
    if state.get("ranked_jobs"):
        print(f"Ranked {len(state['ranked_jobs'])} jobs by relevance")
        print("\nTop ranked jobs:")
        for i, job in enumerate(state['ranked_jobs'][:3], 1):
            print(f"{i}. {job.get('job_title', 'Unknown')} at {job.get('company_name', 'Unknown')} - Score: {job.get('relevance_score', 0)}")
            print(f"   Reason: {job.get('relevance_explanation', '')}")
    if state.get("selected_jobs"):
        print(f"\nSelected {len(state['selected_jobs'])} jobs for CV optimization")
    if state.get("optimized_cvs"):
        print("\nOptimized CVs:")
        for job_key, cv_text in state["optimized_cvs"].items():
            print(f"\n--- Optimized CV for {job_key} ---")
            print(f"Preview (first 300 chars):\n{cv_text[:300]}...")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_automated_test())
