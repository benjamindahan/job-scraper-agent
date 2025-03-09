"""
Main graph definition for the Job Application Assistant.
"""

import os
import asyncio
import json
from typing import Dict, List, Optional, Any, cast
from typing_extensions import TypedDict
import re
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

# Import tool functions
from src.tools.scraping import scrape_jobs
from src.tools.database import init_database, store_jobs, get_unique_locations, filter_jobs
from pydantic import BaseModel, Field

# Setup logging (using print for simplicity)
def log(msg: str):
    print(f"[GRAPH LOG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

# Schema for structured output from the LLM
class CVData(BaseModel):
    """Structured CV data extracted from raw text."""
    skills: List[str] = Field(description="List of technical and soft skills extracted from the CV")
    experience_years: int = Field(description="Total years of professional experience")
    education: str = Field(description="Highest level of education")
    job_titles: List[str] = Field(description="Previous job titles held")
    industries: List[str] = Field(description="Industries the person has worked in")

class JobRelevance(BaseModel):
    """Job relevance score with explanation."""
    score: int = Field(description="Score from 0-100 indicating relevance of job to CV")
    explanation: str = Field(description="Brief explanation of why this score was given")

class UserPreferences(BaseModel):
    """User preferences for job filtering."""
    experience: int = Field(description="Minimum years of experience required")
    date_range: str = Field(description="Time filter (Anytime, Last 24 hours, Last Week, Last Month)")
    locations: List[str] = Field(description="List of acceptable job locations")

# Define our state
# Define our state
class JobScraperState(TypedDict):
    """State for the job scraper graph."""
    # User inputs
    job_title: str
    max_jobs: int
    user_input: Optional[str]  # For user interaction
    preference_input: Optional[str]  # Specifically for preference collection
    job_selection_input: Optional[str]  # Specifically for job selection

    # Scraping results
    job_urls: Optional[List[str]]
    jobs_data: Optional[List[Dict[str, Any]]]

    # User preferences
    user_preferences: Optional[Dict[str, Any]]
    available_locations: Optional[List[str]]
    waiting_for_preferences: Optional[bool]
    last_filter_prefs: Optional[Dict[str, Any]]  # To track if preferences changed
    force_refilter: Optional[bool]  # Flag to force refiltering even if preferences haven't changed

    # Filtered results
    filtered_jobs: Optional[List[Dict[str, Any]]]
    ranked_jobs: Optional[List[Dict[str, Any]]]

    # CV processing
    cv_text: Optional[str]
    cv_data: Optional[Dict[str, Any]]
    waiting_for_cv: Optional[bool]
    cv_file_path: Optional[str]  # Path to the uploaded CV file
    cv_file_name: Optional[str]  # Name of the uploaded CV file
    cv_processed: Optional[bool]  # Flag to track if CV has been processed

    # Job selection
    waiting_for_job_selection: Optional[bool]

    # Output
    selected_jobs: Optional[List[str]]
    optimized_cvs: Optional[Dict[str, str]]

    # Error handling
    error: Optional[str]


# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    log("Warning: OPENAI_API_KEY environment variable not set. Please set it in your .env file or environment variables.")

# LLM initialization
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# NODE FUNCTIONS WITH EXTENSIVE LOGGING

async def search_jobs_node(state: JobScraperState) -> Dict[str, Any]:
    """
    Node to search for jobs based on the job title.

    This node will:
    1. Check if jobs are already in the state
    2. Check if jobs are in the database
    3. Only scrape if necessary
    """
    from src.tools.database import get_jobs_by_title, store_jobs

    log("Entered search_jobs_node with state keys: " + ", ".join(state.keys()))
    job_title = state["job_title"]
    max_jobs = state.get("max_jobs", 20)

    # Check if we already have jobs in the state
    if state.get("jobs_data") and len(state.get("jobs_data", [])) > 0:
        log(f"Already have {len(state['jobs_data'])} jobs in state, skipping search")
        return state

    # Check if we have jobs in the database
    db_jobs = get_jobs_by_title(job_title, max_jobs)

    if db_jobs and len(db_jobs) > 0:
        log(f"Found {len(db_jobs)} jobs in database for '{job_title}'")

        # Make sure all jobs have the search query in searched_job_title
        for job in db_jobs:
            if not job.get('searched_job_title'):
                job['searched_job_title'] = job_title.lower().strip()

        state["jobs_data"] = db_jobs
        state["job_urls"] = [job.get("url", "") for job in db_jobs]
        log("Exiting search_jobs_node with state keys: " + ", ".join(
            state.keys()))
        return state

    # If no jobs in database, scrape them
    log(f"Searching for '{job_title}' jobs (max: {max_jobs})...")

    try:
        from src.tools.scraping import scrape_jobs
        jobs = await scrape_jobs(job_title, max_jobs)

        if not jobs or len(jobs) == 0:
            log("No jobs found through scraping. Checking database as fallback")
            # Final fallback: get any jobs we have
            jobs = get_jobs_by_title("", max_jobs)  # Get any job

            if not jobs:
                log("No jobs found in database either. Returning empty state")
                state[
                    "error"] = "No jobs found. Please try a different search term or try again later."
                return state

        log(f"Scraped {len(jobs)} jobs")
        # Store jobs with the search query
        store_jobs(jobs, job_title.lower().strip())
        state["jobs_data"] = jobs
        state["job_urls"] = [job.get("url", "") for job in jobs]
        log("Exiting search_jobs_node with state keys: " + ", ".join(
            state.keys()))
        return state
    except Exception as e:
        error_msg = f"Error searching for jobs: {str(e)}"
        log(error_msg)
        state["error"] = error_msg
        return state

def prepare_user_preference_prompt(state: JobScraperState) -> Dict[str, Any]:
    log("Entered prepare_user_preferences with state keys: " + ", ".join(state.keys()))
    locations = get_unique_locations()
    log(f"Found {len(locations)} unique locations")
    state["available_locations"] = locations
    state["waiting_for_preferences"] = True
    log("Exiting prepare_user_preferences with state keys: " + ", ".join(state.keys()))
    return state


def collect_user_preferences(state: JobScraperState) -> Dict[str, Any]:
    log("Entered collect_user_preferences with state keys: " + ", ".join(
        state.keys()))
    if not state.get("waiting_for_preferences"):
        log("Not waiting for preferences; exiting collect_user_preferences.")
        return state

    # Use dedicated preference_input field if available, otherwise fall back to user_input
    user_input = state.get("preference_input") or state.get("user_input", "")
    if not user_input:
        log("No user input for preferences; exiting collect_user_preferences.")
        return state

    log(f"Processing user preferences: {user_input}")
    try:
        locations = state.get("available_locations", [])
        location_str = ", ".join(locations)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
            You are an assistant helping parse job preferences. Extract the following information from the user input:

            1. Experience level:
               - If the user mentions specific years (e.g., "6 years"), extract this as a number for min_experience
               - If the user mentions a range (e.g., "2-4 years"), extract min_experience and max_experience
               - If the user mentions "entry level" or "junior", set min_experience to 0 and max_experience to 2
               - If the user mentions "mid-level", set min_experience to 2 and max_experience to 5
               - If the user mentions "senior", set min_experience to 5 with no max_experience
               - Default to min_experience = 0 with no max_experience if unclear

            2. Date range preference:
               - Extract precise date information, handling expressions like:
               - "past 2 weeks" → "Last 2 Weeks"
               - "last month" → "Last Month"
               - "past week" → "Last Week"
               - "recent" or "past few days" → "Last Week"
               - "yesterday" or "24 hours" → "Last 24 hours"
               - "past N days" → convert to appropriate format
               - "past N weeks" → convert to appropriate format
               - Use "Anytime" as default

            3. Preferred locations:
               - If the user specifies locations, find the best matches from this list: {location_str}
               - Use fuzzy matching for locations (e.g., "Tel Aviv" should match "Tel Aviv/Ramat Gan")
               - If the user says "any", "anywhere", or expresses no location preference, return an empty array

            Format your response as a valid JSON object with these fields:
            - min_experience (int): minimum years of experience
            - max_experience (int or null): maximum years of experience (null if no upper limit)
            - date_range (string): specific date range, including custom ranges like "Last 2 Weeks"
            - locations (array of strings): matched locations from the provided list or empty array for no preference
            """),
            HumanMessage(content=user_input)
        ])

        response = llm.invoke(prompt.format_messages())
        log("LLM response for preferences: " + response.content)

        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content)
        json_str = json_match.group(1) if json_match else response.content
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)

        preferences = json.loads(json_str)

        # Validate min_experience (ensure it's an integer)
        if "min_experience" not in preferences or not isinstance(
                preferences["min_experience"], int):
            preferences["min_experience"] = 0

        # Validate max_experience
        if "max_experience" in preferences and not isinstance(
                preferences["max_experience"], int) and preferences[
            "max_experience"] is not None:
            preferences["max_experience"] = None

        # Validate date_range
        if "date_range" not in preferences:
            preferences["date_range"] = "Anytime"

        # Validate and perform fuzzy location matching
        if "locations" not in preferences or not isinstance(
                preferences["locations"], list):
            # Default to no location filter if no valid locations
            preferences["locations"] = []
        else:
            # Fuzzy match locations
            matched_locations = []
            for user_loc in preferences["locations"]:
                user_loc_lower = user_loc.lower()
                for db_loc in locations:
                    if user_loc_lower in db_loc.lower() or db_loc.lower() in user_loc_lower:
                        matched_locations.append(db_loc)
                        break

            # If we found matches, use them; otherwise keep empty for no filtering
            if matched_locations:
                preferences["locations"] = matched_locations

        log(f"Parsed preferences: {preferences}")

        # Store the original search query in preferences for filtering
        if state.get("job_title"):
            preferences["search_query"] = state.get(
                "job_title").lower().strip()

        state["user_preferences"] = preferences
        state["waiting_for_preferences"] = False

        # Clear input fields after use
        state["preference_input"] = None
        state["user_input"] = None

        log("Exiting collect_user_preferences with state keys: " + ", ".join(
            state.keys()))
        return state

    except Exception as e:
        log(f"Error parsing preferences: {e}")
        state["user_preferences"] = {
            "min_experience": 0,
            "max_experience": None,
            "date_range": "Anytime",
            "locations": []  # Default to no location filter on error
        }

        # Store the original search query in preferences for filtering
        if state.get("job_title"):
            state["user_preferences"]["search_query"] = state.get(
                "job_title").lower().strip()

        state["waiting_for_preferences"] = False

        # Clear input fields after use
        state["preference_input"] = None
        state["user_input"] = None

        return state


def filter_jobs_node(state: JobScraperState) -> Dict[str, Any]:
    """
    Node to filter jobs based on user preferences.
    """
    from src.tools.database import filter_jobs, get_all_jobs

    log("Entered filter_jobs_node with state keys: " + ", ".join(state.keys()))

    # Get user preferences or use defaults
    prefs = state.get("user_preferences") or {
        "min_experience": 0,
        "max_experience": None,
        "date_range": "Anytime",
        "locations": [],  # Default to no location filtering
        "search_query": state.get("job_title", "").lower().strip()
        # Include original search query
    }

    log(f"Filtering jobs with preferences: {prefs}")

    try:
        # Avoid redundant filtering if we've already filtered with these preferences
        if (state.get("filtered_jobs") and
                state.get("last_filter_prefs") == prefs and
                not state.get("force_refilter", False)):
            log("Using existing filtered jobs (preferences unchanged)")
            return state

        # Handle the case where we have no jobs data from scraping
        if not state.get("jobs_data"):
            log("No jobs data available for filtering")
            state["filtered_jobs"] = []
            state["waiting_for_cv"] = True
            state["last_filter_prefs"] = prefs.copy()
            return state

        # First try with exact preferences
        filtered = filter_jobs(
            min_experience=prefs.get("min_experience", 0),
            max_experience=prefs.get("max_experience"),
            locations=prefs.get("locations", []),
            date_range=prefs.get("date_range", "Anytime"),
            search_query=prefs.get("search_query")
        )

        # If no results with strict filtering, try more flexible approaches
        if len(filtered) == 0:
            log("No jobs matched strict criteria, trying with only experience filter")

            filtered = filter_jobs(
                min_experience=prefs.get("min_experience", 0),
                max_experience=prefs.get("max_experience"),
                locations=[],  # No location filter
                date_range="Anytime",  # No date filter
                search_query=prefs.get("search_query")
            )

            # If still no results and we have locations, try with just location
            if len(filtered) == 0 and prefs.get("locations"):
                log("No jobs matched experience filter, trying with only location filter")

                filtered = filter_jobs(
                    min_experience=None,  # No experience filter
                    max_experience=None,
                    locations=prefs.get("locations", []),
                    date_range="Anytime",  # No date filter
                    search_query=prefs.get("search_query")
                )

            # If still no results with search query, try without it but keep other filters
            if len(filtered) == 0:
                log("No jobs matched with search query filter, trying without it")

                filtered = filter_jobs(
                    min_experience=prefs.get("min_experience", 0),
                    max_experience=prefs.get("max_experience"),
                    locations=prefs.get("locations", []),
                    date_range=prefs.get("date_range", "Anytime"),
                    search_query=None  # No search query filter
                )

            # If still no results, return all jobs
            if len(filtered) == 0:
                log("No jobs matched any filter criteria, returning all jobs")
                filtered = get_all_jobs()  # Get all jobs from the database

        log(f"Filtered to {len(filtered)} matching jobs")
        state["filtered_jobs"] = filtered
        state[
            "last_filter_prefs"] = prefs.copy()  # Store preferences used for filtering
        state["waiting_for_cv"] = True  # Set to wait for CV if needed
        state["force_refilter"] = False  # Reset force refilter flag

        log("Exiting filter_jobs_node with state keys: " + ", ".join(
            state.keys()))
        return state
    except Exception as e:
        error_msg = f"Error filtering jobs: {str(e)}"
        log(error_msg)
        state["error"] = error_msg
        return state


def collect_cv(state: JobScraperState) -> Dict[str, Any]:
    log("Entered collect_cv with state keys: " + ", ".join(state.keys()))

    # Skip if CV has already been processed
    if state.get("cv_processed") or not state.get("waiting_for_cv"):
        log("CV already processed or not waiting for CV; exiting collect_cv.")
        return state

    # Check for CV file path or user input
    cv_file_path = state.get("cv_file_path")
    cv_file_name = state.get("cv_file_name")
    user_input = state.get("user_input", "")

    cv_text = ""

    # Process CV file if a path is provided
    if cv_file_path:
        log(f"Processing CV file: {cv_file_path}")
        try:
            from src.tools.file_utils import extract_text_from_file

            result = extract_text_from_file(file_path=cv_file_path,
                                            file_name=cv_file_name)

            if result["success"]:
                cv_text = result["text"]
                log(f"Extracted {len(cv_text)} characters from CV file")
            else:
                log(f"Error extracting text from CV file: {result['error']}")
        except Exception as e:
            log(f"Error processing CV file: {e}")
            cv_text = ""

    # Use user_input if no CV file or extraction failed
    if not cv_text and user_input:
        log("Using user input as CV text")
        cv_text = user_input

    # If no CV text is available, exit early
    if not cv_text:
        log("No CV text available; exiting collect_cv.")
        return state

    log("Processing CV...")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
        You are a CV analysis expert. Extract the following information from the provided CV:
        1. List of technical and soft skills
        2. Total years of professional experience (as an integer)
        3. Highest level of education
        4. Previous job titles
        5. Industries worked in

        Format your response as a JSON object with the following structure:
        {
            "skills": ["skill1", "skill2", ...],
            "experience_years": integer,
            "education": "string",
            "job_titles": ["title1", "title2", ...],
            "industries": ["industry1", "industry2", ...]
        }
        """),
        HumanMessage(content=cv_text)
    ])
    try:
        response = llm.invoke(prompt.format_messages())
        log("LLM response for CV analysis: " + response.content)
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content)
        json_str = json_match.group(1) if json_match else response.content
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)
        cv_data = json.loads(json_str)
        log(f"Extracted CV data: {cv_data}")
        state["cv_text"] = cv_text
        state["cv_data"] = cv_data
        state["waiting_for_cv"] = False
        state[
            "cv_processed"] = True  # Mark CV as processed to avoid duplication

        # Clear user_input after use
        state["user_input"] = None

        log("Exiting collect_cv with state keys: " + ", ".join(state.keys()))
        return state
    except Exception as e:
        log(f"Error parsing CV: {e}")
        state["cv_text"] = cv_text
        state["cv_data"] = {
            "skills": [],
            "experience_years": 0,
            "education": "Unknown",
            "job_titles": [],
            "industries": []
        }
        state["waiting_for_cv"] = False
        state[
            "cv_processed"] = True  # Mark as processed even if there was an error

        # Clear user_input after use
        state["user_input"] = None

        return state


def rank_jobs_with_llm(state: JobScraperState) -> Dict[str, Any]:
    log("Entered rank_jobs_with_llm with state keys: " + ", ".join(
        state.keys()))

    # Get filtered jobs or fall back to all jobs if filtering returned nothing
    filtered_jobs = state.get("filtered_jobs", [])
    if not filtered_jobs and state.get("jobs_data"):
        log("No filtered jobs available, using all jobs")
        filtered_jobs = state.get("jobs_data", [])

    cv_data = state.get("cv_data", {})
    cv_text = state.get("cv_text", "")

    log(f"Ranking {len(filtered_jobs)} jobs using LLM...")

    # If no CV provided, return jobs without ranking
    if not cv_text:
        log("No CV provided, returning unranked jobs")
        state["ranked_jobs"] = filtered_jobs
        state["waiting_for_job_selection"] = True
        log("Exiting rank_jobs_with_llm with state keys: " + ", ".join(
            state.keys()))
        return state

    # NEW: Smart re-use of existing rankings
    # If we already have ranked jobs and just filtered the existing set
    if state.get("ranked_jobs") and filtered_jobs:
        # Extract the URLs of filtered jobs
        filtered_urls = {job.get("url", ""): job for job in filtered_jobs}

        # Check if the filtered jobs are already in our ranked list
        existing_ranked_jobs = []
        for job in state.get("ranked_jobs", []):
            job_url = job.get("url", "")
            if job_url in filtered_urls:
                # Use the filtered job data but keep the ranking score and explanation
                ranked_job = filtered_urls[job_url].copy()
                ranked_job["relevance_score"] = job.get("relevance_score", 0)
                ranked_job["relevance_explanation"] = job.get(
                    "relevance_explanation", "")
                existing_ranked_jobs.append(ranked_job)

        # If we found rankings for all filtered jobs, reuse them
        if len(existing_ranked_jobs) == len(filtered_jobs):
            log(f"Reusing existing rankings for all {len(filtered_jobs)} filtered jobs")
            # Sort by relevance score (highest first)
            existing_ranked_jobs.sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True)
            state["ranked_jobs"] = existing_ranked_jobs
            state["waiting_for_job_selection"] = True
            log("Exiting rank_jobs_with_llm with state keys: " + ", ".join(
                state.keys()))
            return state
        # If we found some but not all rankings, continue with normal ranking
        elif existing_ranked_jobs:
            log(f"Found rankings for {len(existing_ranked_jobs)} out of {len(filtered_jobs)} jobs, performing full ranking")

    # Proceed with ranking if we have jobs and CV
    if filtered_jobs:
        ranked_jobs = []
        # UPDATED: Increase batch size from 3 to 5 for efficiency
        batch_size = 5
        for i in range(0, len(filtered_jobs), batch_size):
            batch = filtered_jobs[i:i + batch_size]
            jobs_text = ""
            for idx, job in enumerate(batch):
                jobs_text += f"\nJob {idx + 1}:\n"
                jobs_text += f"Title: {job.get('job_title', 'Unknown')}\n"
                jobs_text += f"Company: {job.get('company_name', 'Unknown')}\n"
                jobs_text += f"Description: {job.get('description', 'No description')[:500]}...\n"
                jobs_text += f"Experience Required: {job.get('experience_text', str(job.get('experience_years', 0)) + ' years')}\n"
                jobs_text += f"Location: {job.get('location', 'Unknown')}\n"
                jobs_text += f"Education: {job.get('education_level', 'Unknown')}\n"
                jobs_text += "---\n"

            cv_skills = ", ".join(cv_data.get("skills", []))
            cv_exp = cv_data.get("experience_years", 0)
            cv_education = cv_data.get("education", "Unknown")
            cv_titles = ", ".join(cv_data.get("job_titles", []))

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=f"""
                You are a job matching expert. Assess how well each job matches the candidate's profile.

                Candidate Profile:
                - Skills: {cv_skills}
                - Experience: {cv_exp} years
                - Education: {cv_education}
                - Previous Titles: {cv_titles}

                For each job, provide:
                1. A relevance score (0-100)
                2. A brief explanation of the score

                Format your response as a valid JSON array of objects, each with:
                - job_index (integer): the job number (1, 2, 3, etc.)
                - score (integer): 0-100 relevance score
                - explanation (string): brief explanation
                Example:
                ```json
                [
                  {{
                    "job_index": 1,
                    "score": 85,
                    "explanation": "Strong match for technical skills and experience"
                  }}
                ]
                ```
                """),
                HumanMessage(
                    content=f"Here are the jobs to evaluate: {jobs_text}")
            ])

            try:
                response = llm.invoke(prompt.format_messages())
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```',
                                       response.content)
                json_str = json_match.group(
                    1) if json_match else response.content
                json_str = re.sub(r'^[^[]*', '', json_str)
                json_str = re.sub(r'[^]]*$', '', json_str)
                rankings = json.loads(json_str)

                for ranking in rankings:
                    job_idx = ranking.get("job_index", 0) - 1
                    if 0 <= job_idx < len(batch):
                        job_copy = batch[job_idx].copy()
                        job_copy["relevance_score"] = ranking.get("score", 0)
                        job_copy["relevance_explanation"] = ranking.get(
                            "explanation", "")
                        ranked_jobs.append(job_copy)

            except Exception as e:
                log(f"Error ranking batch: {e}")
                for job in batch:
                    job_copy = job.copy()
                    job_copy["relevance_score"] = 0
                    job_copy["relevance_explanation"] = "Error in ranking"
                    ranked_jobs.append(job_copy)

        # Sort by relevance score (highest first)
        ranked_jobs.sort(key=lambda x: x.get("relevance_score", 0),
                         reverse=True)
        log(f"Ranked {len(ranked_jobs)} jobs")
        state["ranked_jobs"] = ranked_jobs
    else:
        log("No jobs to rank")
        state["ranked_jobs"] = []

    state["waiting_for_job_selection"] = True
    log("Exiting rank_jobs_with_llm with state keys: " + ", ".join(
        state.keys()))
    return state

def prepare_job_selection(state: JobScraperState) -> Dict[str, Any]:
    log("Entered prepare_job_selection with state keys: " + ", ".join(state.keys()))
    ranked_jobs = state.get("ranked_jobs", [])
    if not ranked_jobs:
        state["waiting_for_job_selection"] = False
    else:
        state["waiting_for_job_selection"] = True
    log("Exiting prepare_job_selection with state keys: " + ", ".join(state.keys()))
    return state


def collect_job_selection(state: JobScraperState) -> Dict[str, Any]:
    log("Entered collect_job_selection with state keys: " + ", ".join(
        state.keys()))

    if not state.get("waiting_for_job_selection"):
        log("Not waiting for job selection; exiting collect_job_selection.")
        return state

    # Use dedicated job_selection_input field if available, otherwise fall back to user_input
    user_input = state.get("job_selection_input") or state.get("user_input",
                                                               "")
    if not user_input:
        log("No user input for job selection; exiting collect_job_selection.")
        return state

    ranked_jobs = state.get("ranked_jobs", [])
    if not ranked_jobs:
        state["selected_jobs"] = []
        state["waiting_for_job_selection"] = False
        return state

    log(f"Processing job selection: {user_input}")

    try:
        job_titles = [
            f"{i + 1}. {job.get('job_title', '')} at {job.get('company_name', '')}"
            for i, job in enumerate(ranked_jobs[:10])]
        job_titles_str = "\n".join(job_titles)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
            You are analyzing a user's job selection from a numbered list. The user has likely provided 
            job numbers or descriptions from this list:
            {job_titles_str}

            YOUR TASK:
            1. Identify which jobs the user wants to select based on their input
            2. Return ONLY a JSON array containing the job numbers (1-based indexing)
            3. For example, if they select jobs 1 and 3, return: [1, 3]

            IMPORTANT INSTRUCTIONS:
            - Return ONLY the JSON array, nothing else - no explanations, no questions
            - Do not include backticks or "json" in your response
            - If the input is ambiguous, select the first job (job #1)
            - Only include valid job numbers from 1 to {len(ranked_jobs[:10])}
            - If the user mentions anything other than job selection, still return a valid JSON array with at least one job
            """),
            HumanMessage(content=user_input)
        ])

        response = llm.invoke(prompt.format_messages())
        log("LLM response for job selection: " + response.content)

        # Clean the response to ensure it's valid JSON
        clean_response = response.content.strip()
        # Remove any backticks, json tags, or explanations
        clean_response = re.sub(r'```json\s*', '', clean_response)
        clean_response = re.sub(r'```', '', clean_response)
        clean_response = re.sub(r'^[^[]*', '', clean_response)
        clean_response = re.sub(r'[^]]*$', '', clean_response)

        selections = json.loads(clean_response)
        selected_jobs = []

        for idx in selections:
            if isinstance(idx, int) and 1 <= idx <= len(ranked_jobs):
                job = ranked_jobs[idx - 1]
                selected_jobs.append(job.get("url", ""))

        # If no valid selections, default to the top job
        if not selected_jobs and len(ranked_jobs) > 0:
            selected_jobs = [ranked_jobs[0].get("url", "")]

        state["selected_jobs"] = selected_jobs
        state["waiting_for_job_selection"] = False

        # Clear input fields after use
        state["job_selection_input"] = None
        state["user_input"] = None

        log(f"Selected {len(selected_jobs)} jobs: {selected_jobs}")
        log("Exiting collect_job_selection with state keys: " + ", ".join(
            state.keys()))
        return state

    except Exception as e:
        log(f"Error parsing job selection: {e}")
        # Default to top job on error
        if ranked_jobs:
            state["selected_jobs"] = [ranked_jobs[0].get("url", "")]
        else:
            state["selected_jobs"] = []

        state["waiting_for_job_selection"] = False

        # Clear input fields after use
        state["job_selection_input"] = None
        state["user_input"] = None

        return state

def optimize_cvs(state: JobScraperState) -> Dict[str, Any]:
    log("Entered optimize_cvs with state keys: " + ", ".join(state.keys()))
    selected_jobs = state.get("selected_jobs", [])
    cv_text = state.get("cv_text", "")
    cv_data = state.get("cv_data", {})
    ranked_jobs = state.get("ranked_jobs", [])
    if not selected_jobs or not cv_text:
        state["optimized_cvs"] = {}
        log("No selected jobs or CV provided; exiting optimize_cvs.")
        return state
    log(f"Optimizing CV for {len(selected_jobs)} jobs...")
    optimized_cvs = {}
    for job_url in selected_jobs:
        job_details = None
        for job in ranked_jobs:
            if job.get("url") == job_url:
                job_details = job
                break
        if not job_details:
            log(f"No job details found for URL: {job_url}")
            continue
        job_title = job_details.get("job_title", "Unknown Position")
        company_name = job_details.get("company_name", "Unknown Company")
        job_description = job_details.get("description", "")
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=f"""
                You are a CV optimization expert. Improve the candidate's CV to better match the job.
                
                Job Details:
                Title: {job_title}
                Company: {company_name}
                Description: {job_description[:1000]}...
                
                Guidelines:
                1. Highlight relevant skills and experiences
                2. Use keywords from the job description
                3. Quantify achievements where possible
                4. Keep the same overall format as the original CV
                5. Maintain the person's actual experience and education (don't fabricate)
                
                Return the optimized CV in full.
                """),
                HumanMessage(content=f"Original CV:\n\n{cv_text}")
            ])
            response = llm.invoke(prompt.format_messages())
            optimized_cv = response.content
            log(f"Optimized CV for {job_title} at {company_name}")
            key = f"{company_name} - {job_title}"
            optimized_cvs[key] = optimized_cv
        except Exception as e:
            log(f"Error optimizing CV for {job_title}: {e}")
    state["optimized_cvs"] = optimized_cvs
    log("Exiting optimize_cvs with state keys: " + ", ".join(state.keys()))
    return state

def finalize_state(state: JobScraperState) -> Dict[str, Any]:
    log("Finalizing state. Current keys: " + ", ".join(state.keys()))
    summary = {
        "jobs_scraped": len(state.get("jobs_data", [])) if state.get("jobs_data") else 0,
        "unique_locations": len(state.get("available_locations", [])) if state.get("available_locations") else 0,
        "filtered_jobs": len(state.get("filtered_jobs", [])) if state.get("filtered_jobs") else 0,
        "ranked_jobs": len(state.get("ranked_jobs", [])) if state.get("ranked_jobs") else 0,
        "selected_jobs": len(state.get("selected_jobs", [])) if state.get("selected_jobs") else 0,
        "optimized_cvs": len(state.get("optimized_cvs", {})) if state.get("optimized_cvs") else 0,
    }
    state["state"] = summary
    log("Final state finalized with keys: " + ", ".join(state.keys()))
    return state

def initial_state_creator() -> JobScraperState:
    log("Creating initial state...")
    state = {
        "job_title": "",
        "max_jobs": 20,
        "user_input": None,
        "preference_input": None,
        "job_selection_input": None,
        "job_urls": None,
        "jobs_data": None,
        "user_preferences": None,
        "available_locations": None,
        "waiting_for_preferences": None,
        "last_filter_prefs": None,
        "force_refilter": False,
        "filtered_jobs": None,
        "ranked_jobs": None,
        "cv_text": None,
        "cv_data": None,
        "waiting_for_cv": None,
        "cv_file_path": None,
        "cv_file_name": None,
        "cv_processed": False,
        "waiting_for_job_selection": None,
        "selected_jobs": None,
        "optimized_cvs": None,
        "error": None
    }
    log("Initial state created with keys: " + ", ".join(state.keys()))
    return state


def should_wait_for_cv(state: JobScraperState) -> str:
    log("Evaluating should_wait_for_cv with state keys: " + ", ".join(
        state.keys()))

    # If CV is already processed, skip to ranking
    if state.get("cv_processed"):
        return "rank_jobs"

    # If we already have CV data
    if state.get("cv_data") is not None:
        return "rank_jobs"

    # If we have a CV file path but haven't processed it yet, process it
    if state.get("cv_file_path") and not state.get("cv_text"):
        return "collect_cv"

    # If we've explicitly been told to wait for CV and have user input (CV content)
    if state.get("waiting_for_cv", False) and (
            state.get("user_input") is not None or state.get(
            "cv_file_path") is not None):
        return "collect_cv"

    # Default: proceed to ranking (possibly without CV)
    return "rank_jobs"


# Update the conditional routing function for user preferences
def should_wait_for_preferences(state: JobScraperState) -> str:
    log("Evaluating should_wait_for_preferences with state keys: " + ", ".join(
        state.keys()))

    # If we explicitly have preference input, collect preferences
    if state.get("preference_input") is not None:
        return "collect_user_preferences"

    # If we're explicitly waiting for preferences and have user input
    if state.get("waiting_for_preferences", False) and state.get(
            "user_input") is not None:
        return "collect_user_preferences"

    # If we already have preferences, skip to filtering
    if state.get("user_preferences") is not None:
        return "filter_jobs"

    # Default: proceed to filtering with default preferences
    return "filter_jobs"


# Update the conditional routing function for job selection
def should_wait_for_job_selection(state: JobScraperState) -> str:
    log("Evaluating should_wait_for_job_selection with state keys: " + ", ".join(
        state.keys()))

    # If we explicitly have job selection input, collect the selection
    if state.get("job_selection_input") is not None:
        return "collect_job_selection"

    # If we're waiting for job selection and have user input
    if state.get("waiting_for_job_selection", False) and state.get(
            "user_input") is not None:
        return "collect_job_selection"

    # If we're not waiting or already have selected jobs
    if not state.get("waiting_for_job_selection", False) or state.get(
            "selected_jobs") is not None:
        return "optimize_cvs"

    # Default: proceed to optimization (might be without any selected jobs)
    return "optimize_cvs"

# Build the graph
def build_graph():
    log("Building the graph...")
    graph = StateGraph(JobScraperState)
    graph.add_node("search_jobs", search_jobs_node)
    graph.add_node("prepare_user_preferences", prepare_user_preference_prompt)
    graph.add_node("collect_user_preferences", collect_user_preferences)
    graph.add_node("filter_jobs", filter_jobs_node)
    graph.add_node("collect_cv", collect_cv)
    graph.add_node("rank_jobs", rank_jobs_with_llm)
    graph.add_node("prepare_job_selection", prepare_job_selection)
    graph.add_node("collect_job_selection", collect_job_selection)
    graph.add_node("optimize_cvs", optimize_cvs)
    graph.add_node("finalize_state", finalize_state)
    graph.add_node("wait_for_input", lambda x: x)  # No-op node for waiting

    graph.set_entry_point("search_jobs")
    graph.add_edge("search_jobs", "prepare_user_preferences")
    graph.add_conditional_edges(
        "prepare_user_preferences",
        should_wait_for_preferences,
        {
            "wait_for_input": "wait_for_input",
            "collect_user_preferences": "collect_user_preferences",
            "filter_jobs": "filter_jobs"
        }
    )
    graph.add_edge("collect_user_preferences", "filter_jobs")
    graph.add_conditional_edges(
        "filter_jobs",
        should_wait_for_cv,
        {
            "wait_for_input": "wait_for_input",
            "collect_cv": "collect_cv",
            "rank_jobs": "rank_jobs"
        }
    )
    graph.add_edge("collect_cv", "rank_jobs")
    graph.add_edge("rank_jobs", "prepare_job_selection")
    graph.add_conditional_edges(
        "prepare_job_selection",
        should_wait_for_job_selection,
        {
            "wait_for_input": "wait_for_input",
            "collect_job_selection": "collect_job_selection",
            "optimize_cvs": "optimize_cvs"
        }
    )
    graph.add_edge("collect_job_selection", "optimize_cvs")
    graph.add_edge("optimize_cvs", "finalize_state")
    graph.add_edge("finalize_state", END)
    log("Graph built successfully.")
    return graph.compile()

# Create the graph
graph = build_graph()