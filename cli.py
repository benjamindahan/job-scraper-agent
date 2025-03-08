import argparse
import asyncio
import sys
import os
import time
from typing import Optional, List, Dict, Any

from src.graph import graph, initial_state_creator
from src.tools.file_utils import extract_text_from_file
from src.tools.database import get_jobs_by_title, get_all_jobs


async def run_agent(job_title: str, max_jobs: int = 20,
                    cv_file_path: Optional[str] = None,
                    use_db_only: bool = False):
    """
    Run the job scraper agent with the specified parameters.
    Following the flow:
    1. Get jobs for the given title (from database if use_db_only=True, otherwise try scraping)
    2. Ask for user preferences
    3. Filter jobs based on preferences
    4. Process CV and rank jobs
    5. Let user select jobs
    6. Optimize CV for selected jobs

    Args:
        job_title: Job title to search for
        max_jobs: Maximum number of jobs to scrape
        cv_file_path: Path to the CV file (optional)
        use_db_only: Whether to use only the database (no scraping)
    """
    print(f"\n=== Job Search for '{job_title}' ===\n")

    # Step 1: Initial state setup and job data retrieval
    state = initial_state_creator()
    state["job_title"] = job_title
    state["max_jobs"] = max_jobs

    # Use database-only mode if specified or if we've been rate-limited recently
    if use_db_only:
        print("Using database only (no scraping)...")
        db_jobs = get_jobs_by_title(job_title, max_jobs)

        if not db_jobs or len(db_jobs) == 0:
            print(f"No jobs found in database for '{job_title}'.")
            print("Trying to find any available jobs...")
            db_jobs = get_all_jobs()

            if not db_jobs or len(db_jobs) == 0:
                print("No jobs found in the database. Please try again later.")
                return

        print(f"Found {len(db_jobs)} jobs in database.")
        state["jobs_data"] = db_jobs
        state["job_urls"] = [job.get("url", "") for job in db_jobs]
    else:
        # Normal flow - will try scraping first, fall back to database
        print("Scraping job listings... (this may take a moment)")
        config = {"configurable": {"thread_id": "cli-thread"}}

        # Run just the job search part of the graph
        state = await graph.ainvoke(state, config=config)

        # Check for errors or no jobs
        if state.get("error") or not state.get("jobs_data") or len(
                state.get("jobs_data", [])) == 0:
            if state.get("error"):
                print(f"Error: {state['error']}")
            else:
                print("No jobs found through scraping. Checking database...")

            # Try to get jobs from database instead
            db_jobs = get_jobs_by_title(job_title, max_jobs)

            if not db_jobs or len(db_jobs) == 0:
                print(f"No jobs found in database for '{job_title}' either.")
                print("Trying to find any available jobs...")

                # Last resort - get any jobs
                db_jobs = get_all_jobs()

                if not db_jobs or len(db_jobs) == 0:
                    print(
                        "No jobs found. Please try a different search term or try again later.")
                    return

            print(f"Found {len(db_jobs)} jobs in database.")
            state["jobs_data"] = db_jobs
            state["job_urls"] = [job.get("url", "") for job in db_jobs]

    # Step 2: Get user preferences
    available_locations = state.get("available_locations", [])
    if not available_locations:
        # Get locations directly from database if not in state
        from src.tools.database import get_unique_locations
        available_locations = get_unique_locations()
        state["available_locations"] = available_locations

    if available_locations:
        location_display = ", ".join(available_locations[:5])
        if len(available_locations) > 5:
            location_display += f" and {len(available_locations) - 5} more"
        print(f"\nAvailable locations: {location_display}")

    print("\n=== Job Search Preferences ===")
    print("Please enter your preferences:")
    print("1. Experience level (e.g. 'entry level', '3 years', 'senior')")
    print("2. Preferred location(s) or 'any' if no preference")
    print("3. Date range (e.g. 'last week', 'last month', 'anytime')")
    print(
        "\nExample: '3 years experience in Tel Aviv, posted in the last week'")

    user_prefs = input("\nYour preferences: ")

    # Step 3: Filter jobs with user preferences
    print("\nFiltering jobs based on your preferences...")
    state["user_input"] = user_prefs
    state["waiting_for_preferences"] = True

    config = {"configurable": {"thread_id": "cli-thread"}}
    state = await graph.ainvoke(state, config=config)

    filtered_jobs = state.get("filtered_jobs", [])
    if not filtered_jobs or len(filtered_jobs) == 0:
        print(
            "\n⚠️ No jobs match your exact criteria. Showing all jobs instead.")
        filtered_jobs = state.get("jobs_data", [])
        state["filtered_jobs"] = filtered_jobs
    else:
        print(f"\n✅ Found {len(filtered_jobs)} jobs matching your preferences")

    # Step 4: Process CV if provided
    if cv_file_path:
        if not os.path.exists(cv_file_path):
            print(f"\n❌ Error: CV file not found at {cv_file_path}")
            return

        print(f"\n=== Processing CV ===")
        print(f"Reading file: {cv_file_path}")

        result = extract_text_from_file(file_path=cv_file_path)
        if not result["success"]:
            print(f"\n❌ Error extracting text from CV file: {result['error']}")
            return

        cv_text = result["text"]
        print(f"✅ Successfully extracted {len(cv_text)} characters from CV")

        # Process the CV and rank jobs
        print("\nRanking jobs based on CV relevance...")
        state["cv_file_path"] = cv_file_path
        state["user_input"] = cv_text
        state["waiting_for_cv"] = True
        state[
            "waiting_for_preferences"] = False  # Ensure we don't reprocess preferences

        state = await graph.ainvoke(state, config=config)

    # Step 5: Display ranked jobs and get user selection
    ranked_jobs = state.get("ranked_jobs", [])
    if not ranked_jobs:
        ranked_jobs = filtered_jobs

    if not ranked_jobs or len(ranked_jobs) == 0:
        print("\n❌ No matching jobs found to display.")
        return

    print("\n=== Top Matching Jobs ===")
    display_count = min(10, len(ranked_jobs))

    for i, job in enumerate(ranked_jobs[:display_count], 1):
        print(
            f"\n{i}. {job.get('job_title', 'Unknown Position')} at {job.get('company_name', 'Unknown Company')}")
        if job.get('location'):
            print(f"   Location: {job.get('location')}")
        if job.get('experience_years') or job.get('experience_text'):
            print(
                f"   Experience: {job.get('experience_text', str(job.get('experience_years', '')) + ' years')}")
        if job.get('relevance_score'):
            print(f"   Match Score: {job.get('relevance_score')}/100")
            print(f"   Reason: {job.get('relevance_explanation', 'N/A')}")

    # Step 6: User job selection
    job_selection = input(
        "\nSelect job numbers to optimize your CV for (e.g., '1,3'): ")

    state["user_input"] = job_selection
    state["waiting_for_job_selection"] = True
    state["waiting_for_cv"] = False  # Ensure we don't reprocess CV

    state = await graph.ainvoke(state, config=config)

    # Step 7: Display results
    print("\n=== Results ===")

    if state.get("selected_jobs"):
        print(
            f"Selected {len(state.get('selected_jobs', []))} jobs for CV optimization")

    if state.get("optimized_cvs") and len(state.get("optimized_cvs", {})) > 0:
        print("\n=== Optimized CVs ===")
        for job_key, cv_text in state.get("optimized_cvs", {}).items():
            print(f"\n• {job_key}")

            # Optionally save to files
            safe_filename = job_key.replace(" ", "_").replace("/",
                                                              "_").replace(
                "\\", "_")
            filename = f"optimized_cv_{safe_filename}.txt"

            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(cv_text)
                print(f"  Saved to: {filename}")
            except Exception as e:
                print(f"  Could not save to file: {str(e)}")
                print("  Preview:")
                print(f"  {cv_text[:200]}...")
    else:
        print("\nNo CV optimizations were created. This might be because:")
        print("- No CV was provided")
        print("- No jobs were selected")
        print("- There was an error processing the selected jobs")


def main():
    parser = argparse.ArgumentParser(description="Job Scraper Agent CLI")
    parser.add_argument("job_title", help="Job title to search for")
    parser.add_argument("--max-jobs", type=int, default=20,
                        help="Maximum number of jobs to scrape")
    parser.add_argument("--cv", help="Path to the CV file (PDF, DOCX, or TXT)")
    parser.add_argument("--db-only", action="store_true",
                        help="Use only database, no scraping (faster)")

    args = parser.parse_args()

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(
        run_agent(args.job_title, args.max_jobs, args.cv, args.db_only))


if __name__ == "__main__":
    main()