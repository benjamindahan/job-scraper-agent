"""
Main graph definition for the Job Application Assistant.
"""

import os
import asyncio
from typing import Annotated, Dict, List, Optional, Any
from typing_extensions import TypedDict

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Import tool functions
from tools.scraping import scrape_jobs
from tools.database import init_database, store_jobs, get_unique_locations, filter_jobs

# Define our state
class JobScraperState(TypedDict):
    """State for the job scraper graph."""
    # User inputs
    job_title: str
    max_jobs: int

    # Scraping results
    job_urls: Optional[List[str]]
    jobs_data: Optional[List[Dict[str, Any]]]

    # User preferences
    user_preferences: Optional[Dict[str, Any]]

    # Filtered results
    filtered_jobs: Optional[List[Dict[str, Any]]]
    ranked_jobs: Optional[List[Dict[str, Any]]]

    # CV processing
    cv_text: Optional[str]
    cv_data: Optional[Dict[str, Any]]

    # Output
    selected_jobs: Optional[List[str]]
    optimized_cvs: Optional[Dict[str, str]]

    # Error handling
    error: Optional[str]

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("Please set it in your .env file or environment variables.")

# LLM initialization
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Node functions
async def search_jobs_node(state: JobScraperState) -> Dict[str, Any]:
    """Node to search for jobs based on the job title."""
    job_title = state["job_title"]
    max_jobs = state.get("max_jobs", 20)

    print(f"Searching for '{job_title}' jobs (max: {max_jobs})...")

    try:
        # Use the scraper to fetch jobs
        jobs = await scrape_jobs(job_title, max_jobs)
        print(f"Scraped {len(jobs)} jobs")

        # Store jobs in database
        store_jobs(jobs)

        return {
            "jobs_data": jobs,
            "job_urls": [job.get("url", "") for job in jobs]
        }
    except Exception as e:
        error_msg = f"Error searching for jobs: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

def get_user_preferences(state: JobScraperState) -> Dict[str, Any]:
    """Node to get user preferences for filtering jobs."""
    # In a real implementation, this would prompt the user for preferences
    # For now, return default preferences (can be updated via user input later)

    print("Getting available job locations...")

    # Get unique locations from the database for user selection
    locations = get_unique_locations()
    print(f"Found {len(locations)} unique locations")

    # Default preferences
    preferences = {
        "experience": 0,  # Default: Any experience level
        "date_range": "Anytime",  # Default: Any time
        "locations": locations[:3] if locations else []  # Default: First 3 locations
    }

    print(f"Using preferences: {preferences}")

    return {
        "user_preferences": preferences
    }

def filter_jobs_node(state: JobScraperState) -> Dict[str, Any]:
    """Node to filter jobs based on user preferences."""
    prefs = state["user_preferences"]

    print(f"Filtering jobs with preferences: {prefs}")

    try:
        filtered = filter_jobs(
            min_experience=prefs.get("experience", 0),
            locations=prefs.get("locations", []),
            date_range=prefs.get("date_range", "Anytime")
        )

        print(f"Filtered to {len(filtered)} matching jobs")

        return {"filtered_jobs": filtered}
    except Exception as e:
        error_msg = f"Error filtering jobs: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

def rank_jobs(state: JobScraperState) -> Dict[str, Any]:
    """Node to rank jobs by relevance to the user's CV."""
    filtered_jobs = state["filtered_jobs"]
    cv_text = state.get("cv_text", "")

    print(f"Ranking {len(filtered_jobs)} jobs...")

    # If no CV provided, return unranked jobs
    if not cv_text:
        print("No CV provided, returning unranked jobs")
        return {"ranked_jobs": filtered_jobs}

    # In a real implementation, this would analyze the CV and rank jobs
    # For now, use a simple ranking based on job title

    ranked_jobs = filtered_jobs.copy()

    # Sort alphabetically as a placeholder (should be replaced with real ranking)
    ranked_jobs.sort(key=lambda job: job.get("job_title", ""))

    print(f"Ranked {len(ranked_jobs)} jobs")

    return {"ranked_jobs": ranked_jobs}

def select_jobs(state: JobScraperState) -> Dict[str, Any]:
    """Node to let the user select jobs for CV optimization."""
    # In a real implementation, this would prompt the user to select jobs
    # For now, select the top 2 jobs if available

    ranked_jobs = state["ranked_jobs"]

    print(f"Selecting from {len(ranked_jobs)} ranked jobs...")

    selected = ranked_jobs[:2] if len(ranked_jobs) >= 2 else ranked_jobs
    selected_urls = [job.get("url", "") for job in selected]

    print(f"Selected {len(selected_urls)} jobs for CV optimization")

    return {
        "selected_jobs": selected_urls
    }

# Build the graph
def build_graph():
    """Build and return the job application assistant graph."""
    graph = StateGraph(JobScraperState)

    # Add nodes
    graph.add_node("search_jobs", search_jobs_node)
    graph.add_node("get_user_preferences", get_user_preferences)
    graph.add_node("filter_jobs", filter_jobs_node)
    graph.add_node("rank_jobs", rank_jobs)
    graph.add_node("select_jobs", select_jobs)

    # Add edges
    graph.set_entry_point("search_jobs")
    graph.add_edge("search_jobs", "get_user_preferences")
    graph.add_edge("get_user_preferences", "filter_jobs")
    graph.add_edge("filter_jobs", "rank_jobs")
    graph.add_edge("rank_jobs", "select_jobs")
    graph.add_edge("select_jobs", END)

    return graph.compile()

# Create the graph
graph = build_graph()

# Test execution function (for development)
async def main():
    """Test the graph with a sample job title."""
    print("\n=== Testing Job Application Assistant Graph ===\n")

    result = await graph.ainvoke({
        "job_title": "data scientist",
        "max_jobs": 10
    })

    print("\n=== Results ===")
    print(f"Found and processed {len(result.get('jobs_data', []))} jobs")

    if result.get('filtered_jobs'):
        print(f"Filtered to {len(result['filtered_jobs'])} matching jobs")

    if result.get('selected_jobs'):
        print(f"Selected {len(result['selected_jobs'])} jobs for CV optimization")

    # Print some selected job details
    if result.get('ranked_jobs'):
        print("\nTop ranked jobs:")
        for i, job in enumerate(result['ranked_jobs'][:3], 1):
            print(f"{i}. {job['job_title']} at {job['company_name']}")

    if result.get('error'):
        print(f"\nError encountered: {result['error']}")

    return result

# Run test if called directly
if __name__ == "__main__":
    import sys

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())