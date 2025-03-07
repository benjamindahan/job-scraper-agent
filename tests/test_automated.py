"""
Automated tests for the Job Application Assistant.
"""

import asyncio
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph import graph, initial_state_creator


async def run_automated_test():
    """Run an automated test of the job application assistant."""
    print("\n=== Automated Test: Job Application Assistant ===\n")

    # Initial state
    state = initial_state_creator()
    state["job_title"] = "data scientist"
    state["max_jobs"] = 10

    print(
        f"1. Searching for '{state['job_title']}' jobs (max: {state['max_jobs']})...")

    # Create a persistent thread for the test
    config = {"configurable": {"thread_id": "test-thread"}}

    try:
        # Start the graph - use ainvoke instead of invoke
        result = await graph.ainvoke(state, config=config)
        current_state = result["state"]

        # Process preferences
        if current_state.get("waiting_for_preferences"):
            print("\n2. Setting job preferences automatically...")
            current_state[
                "user_input"] = "I'm looking for jobs with at least 3 years of experience, posted in the Last Month, in Tel Aviv"
            # Use ainvoke here too
            result = await graph.ainvoke(current_state, config=config)
            current_state = result["state"]

        # Process CV input
        if current_state.get("waiting_for_cv"):
            print("\n3. Providing sample CV...")
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
            # Use ainvoke here too
            result = await graph.ainvoke(current_state, config=config)
            current_state = result["state"]

        # Process job selection
        if current_state.get("waiting_for_job_selection"):
            print("\n4. Selecting top jobs automatically...")
            current_state["user_input"] = "1, 2"
            # Use ainvoke here too
            result = await graph.ainvoke(current_state, config=config)
            current_state = result["state"]

        # Display results
        print_results(current_state)
        return current_state

    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def print_results(state):
    """Print the results of the test."""
    print("\n=== Test Results ===")

    if state.get("error"):
        print(f"Error encountered: {state['error']}")
        return

    # Print job statistics
    if state.get("jobs_data"):
        print(f"Found and processed {len(state['jobs_data'])} jobs")

    if state.get("filtered_jobs"):
        print(f"Filtered to {len(state['filtered_jobs'])} matching jobs")

    if state.get("ranked_jobs"):
        print(f"Ranked {len(state['ranked_jobs'])} jobs by relevance")

        # Show top ranked jobs
        print("\nTop ranked jobs:")
        for i, job in enumerate(state['ranked_jobs'][:3], 1):
            print(
                f"{i}. {job['job_title']} at {job['company_name']} - Score: {job.get('relevance_score', 0)}")
            print(f"   Reason: {job.get('relevance_explanation', '')}")

    if state.get("selected_jobs"):
        print(
            f"\nSelected {len(state['selected_jobs'])} jobs for CV optimization")

    # Print optimized CVs
    if state.get("optimized_cvs"):
        print("\nOptimized CVs:")
        for job_key, cv_text in state["optimized_cvs"].items():
            print(f"\n--- Optimized CV for {job_key} ---")
            print(f"Preview (first 300 chars):\n{cv_text[:300]}...")


if __name__ == "__main__":
    # Handle Windows event loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_automated_test())