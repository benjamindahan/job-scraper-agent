#!/usr/bin/env python3
"""
Interactive CLI for the Job Application Assistant.
"""
import argparse
import asyncio
import sys
import os
from typing import Dict, Any, Optional

from src.graph import graph, initial_state_creator
from src.cv_upload import process_cv_file_and_invoke_graph


async def run_agent(job_title: str, max_jobs: int = 20,
                    cv_file_path: Optional[str] = None):
    """
    Run the job scraper agent with the specified parameters.

    Args:
        job_title: Job title to search for
        max_jobs: Maximum number of jobs to scrape
        cv_file_path: Path to the CV file (optional)
    """
    print(f"Starting job search for '{job_title}'...")

    # Initial state
    state = initial_state_creator()
    state["job_title"] = job_title
    state["max_jobs"] = max_jobs

    # If a CV file is provided, process it first
    if cv_file_path:
        if not os.path.exists(cv_file_path):
            print(f"Error: CV file not found at {cv_file_path}")
            return

        print(f"Processing CV file: {cv_file_path}")
        state = await process_cv_file_and_invoke_graph(file_path=cv_file_path,
                                                       current_state=state)

        if 'error' in state:
            print(f"Error processing CV file: {state['error']}")
            return
    else:
        # No CV file, invoke the graph normally
        config = {"configurable": {"thread_id": "cli-thread"}}
        state = await graph.ainvoke(state, config=config)

    # Print results
    print("\n=== Results ===")
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
            print(
                f"{i}. {job.get('job_title', 'Unknown')} at {job.get('company_name', 'Unknown')}")
            if 'relevance_score' in job:
                print(f"   Score: {job['relevance_score']}")
                print(f"   Reason: {job.get('relevance_explanation', '')}")

    if state.get("selected_jobs"):
        print(
            f"\nSelected {len(state['selected_jobs'])} jobs for CV optimization")

    if state.get("optimized_cvs"):
        print("\nOptimized CVs:")
        for job_key in state["optimized_cvs"]:
            print(f"- {job_key}")

async def run_interactive_session():
    """Run an interactive session with the job application assistant."""
    print("\n=== Job Application Assistant ===\n")
    print(
        "This tool helps you find relevant jobs and optimize your CV for them.")

    # Initial query
    job_title = input("\nEnter job title to search for: ")
    max_jobs = input("Maximum number of jobs to scrape (5-20) [default: 10]: ")

    # Process max_jobs input
    try:
        max_jobs = int(max_jobs) if max_jobs.strip() else 10
        max_jobs = max(5, min(20, max_jobs))
    except ValueError:
        max_jobs = 10
        print("Invalid number, using default of 10 jobs.")

    print(f"\nSearching for '{job_title}' jobs (max: {max_jobs})...")

    # Initial state
    state = initial_state_creator()
    state["job_title"] = job_title
    state["max_jobs"] = max_jobs

    # Create a persistent thread for the session
    config = {"configurable": {"thread_id": "interactive-session"}}

    try:
        # Start the graph
        result = await graph.ainvoke(state, config=config)
        current_state = result["state"]

        # Process the graph in a loop
        while not result.get("end"):
            if current_state.get("waiting_for_preferences"):
                current_state = await handle_preferences_input(current_state)
            elif current_state.get("waiting_for_cv"):
                current_state = await handle_cv_input(current_state)
            elif current_state.get("waiting_for_job_selection"):
                current_state = await handle_job_selection(current_state)
            else:
                # Continue with empty input
                current_state["user_input"] = None

            # Continue the graph
            result = await graph.ainvoke(current_state, config=config)
            current_state = result["state"]

        # Display final results
        print_final_results(current_state)

    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()


async def handle_preferences_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle user input for job preferences."""
    print("\n--- Job Preferences ---")
    print("Available locations:",
          ', '.join(state.get("available_locations", [])))

    exp = input("Minimum years of experience: ")

    # Print date range options
    print("\nDate range options:")
    print("1. Anytime")
    print("2. Last 24 hours")
    print("3. Last Week")
    print("4. Last Month")
    date_choice = input("Choose a date range (1-4): ")

    # Map choice to value
    date_map = {
        "1": "Anytime",
        "2": "Last 24 hours",
        "3": "Last Week",
        "4": "Last Month"
    }
    date_range = date_map.get(date_choice, "Anytime")

    # Location selection
    locations = input("Preferred locations (comma-separated): ")

    # Construct natural language input
    user_input = (
        f"I'm looking for jobs with at least {exp} years of experience, "
        f"posted {date_range}, in these locations: {locations}"
    )

    state["user_input"] = user_input
    return state


async def handle_cv_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle user input for CV."""
    print("\n--- CV Input ---")
    print("You can enter your CV text or use our sample CV.")

    use_sample = input("Use sample CV? (y/n) [default: y]: ").lower()

    if use_sample != "n":
        # Use sample CV
        print("Using sample CV...")
        cv_text = """
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
    else:
        # Get user's CV
        print("\nEnter your CV text (type END on a new line when finished):")
        lines = []
        while True:
            line = input()
            if line == "END":
                break
            lines.append(line)
        cv_text = "\n".join(lines)

    state["user_input"] = cv_text
    return state


async def handle_job_selection(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle user selection of jobs for CV optimization."""
    print("\n--- Job Selection for CV Optimization ---")

    ranked_jobs = state.get("ranked_jobs", [])
    if not ranked_jobs:
        print("No jobs available for selection.")
        state["user_input"] = ""
        return state

    print("Top ranked jobs:")
    for i, job in enumerate(ranked_jobs[:10], 1):
        score = job.get("relevance_score", 0)
        print(
            f"{i}. {job['job_title']} at {job['company_name']} - Score: {score}")
        print(f"   Location: {job.get('location', 'Unknown')}")
        print(f"   Experience: {job.get('experience_years', 0)} years")
        print()

    selection = input(
        "Enter job numbers to optimize CV for (comma-separated): ")
    state["user_input"] = selection
    return state


def print_final_results(state: Dict[str, Any]) -> None:
    """Print the final results."""
    print("\n=== Results ===")

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

    if state.get("selected_jobs"):
        print(
            f"Selected {len(state['selected_jobs'])} jobs for CV optimization")

    # Print optimized CVs
    if state.get("optimized_cvs"):
        print("\nOptimized CVs:")
        for job_key, cv_text in state["optimized_cvs"].items():
            print(f"\n--- Optimized CV for {job_key} ---")
            print(f"Preview (first 300 chars):")
            print(cv_text[:300])
            print("...")

            # Offer to save to file
            save = input(f"\nSave this CV to file? (y/n): ").lower()
            if save == "y":
                filename = f"optimized_cv_{job_key.replace(' ', '_')}.txt"
                with open(filename, "w") as f:
                    f.write(cv_text)
                print(f"Saved to {filename}")

    print("\nThank you for using the Job Application Assistant!")


def main():
    parser = argparse.ArgumentParser(description="Job Scraper Agent CLI")
    parser.add_argument("job_title", help="Job title to search for")
    parser.add_argument("--max-jobs", type=int, default=20,
                        help="Maximum number of jobs to scrape")
    parser.add_argument("--cv", help="Path to the CV file (PDF, DOCX, or TXT)")

    args = parser.parse_args()

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_agent(args.job_title, args.max_jobs, args.cv))


if __name__ == "__main__":
    main()