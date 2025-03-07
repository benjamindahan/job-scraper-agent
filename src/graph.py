"""
Main graph definition for the Job Application Assistant.
"""

import os
import asyncio
import json
from typing import Annotated, Dict, List, Optional, Any, cast
from typing_extensions import TypedDict
import re
from datetime import datetime

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
class JobScraperState(TypedDict):
    """State for the job scraper graph."""
    # User inputs
    job_title: str
    max_jobs: int
    user_input: Optional[str]  # For user interaction

    # Scraping results
    job_urls: Optional[List[str]]
    jobs_data: Optional[List[Dict[str, Any]]]

    # User preferences
    user_preferences: Optional[Dict[str, Any]]
    available_locations: Optional[List[str]]
    waiting_for_preferences: Optional[bool]

    # Filtered results
    filtered_jobs: Optional[List[Dict[str, Any]]]
    ranked_jobs: Optional[List[Dict[str, Any]]]

    # CV processing
    cv_text: Optional[str]
    cv_data: Optional[Dict[str, Any]]
    waiting_for_cv: Optional[bool]

    # Job selection
    waiting_for_job_selection: Optional[bool]

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

def prepare_user_preference_prompt(state: JobScraperState) -> Dict[str, Any]:
    """Prepare the state for collecting user preferences."""
    print("Getting available job locations...")

    # Get unique locations from the database for user selection
    locations = get_unique_locations()
    print(f"Found {len(locations)} unique locations")

    # Prepare for user input
    return {
        "available_locations": locations,
        "waiting_for_preferences": True
    }

def collect_user_preferences(state: JobScraperState) -> Dict[str, Any]:
    """Collect user preferences for job filtering."""
    if not state.get("waiting_for_preferences"):
        return {}

    user_input = state.get("user_input", "")
    if not user_input:
        return {}

    print(f"Processing user preferences: {user_input}")

    # Parse user input
    try:
        # Use the LLM to parse unstructured preferences
        locations = state.get("available_locations", [])
        location_str = ", ".join(locations)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
            You are an assistant helping parse job preferences. Extract the following information:
            1. Years of experience (as an integer)
            2. Date range preference (one of: "Anytime", "Last 24 hours", "Last Week", "Last Month")
            3. Preferred locations from this list: {location_str}
            
            Format your response as a valid JSON object with fields: 
            - experience (int)
            - date_range (string)
            - locations (array of strings)
            """
            ),
            HumanMessage(content=user_input)
        ])

        response = llm.invoke(prompt)

        # Extract JSON from response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content

        # Try to clean up any non-JSON content
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)

        preferences = json.loads(json_str)

        # Validate preferences
        if "experience" not in preferences or not isinstance(preferences["experience"], int):
            preferences["experience"] = 0

        if "date_range" not in preferences:
            preferences["date_range"] = "Anytime"
        elif preferences["date_range"] not in ["Anytime", "Last 24 hours", "Last Week", "Last Month"]:
            preferences["date_range"] = "Anytime"

        if "locations" not in preferences or not isinstance(preferences["locations"], list):
            preferences["locations"] = locations[:3] if locations else []
        else:
            # Validate each location
            valid_locations = []
            for loc in preferences["locations"]:
                if loc in locations:
                    valid_locations.append(loc)
            preferences["locations"] = valid_locations if valid_locations else (locations[:3] if locations else [])

        print(f"Parsed preferences: {preferences}")

        return {
            "user_preferences": preferences,
            "waiting_for_preferences": False  # Done collecting preferences
        }
    except Exception as e:
        print(f"Error parsing preferences: {e}")
        # Use default preferences on error
        return {
            "user_preferences": {
                "experience": 0,
                "date_range": "Anytime",
                "locations": locations[:3] if locations else []
            },
            "waiting_for_preferences": False
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

        return {
            "filtered_jobs": filtered,
            "waiting_for_cv": True  # Next step is to collect CV
        }
    except Exception as e:
        error_msg = f"Error filtering jobs: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

def collect_cv(state: JobScraperState) -> Dict[str, Any]:
    """Collect and parse the user's CV."""
    if not state.get("waiting_for_cv"):
        return {}

    user_input = state.get("user_input", "")
    if not user_input:
        return {}

    print("Processing CV...")

    # For now, we'll use the user input directly as the CV text
    # In a real application, this would parse a PDF/DOCX file
    cv_text = user_input

    # Use LLM to extract structured data from CV
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
        """
        ),
        HumanMessage(content=cv_text)
    ])

    try:
        response = llm.invoke(prompt)

        # Extract JSON from response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content

        # Try to clean up any non-JSON content
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)

        cv_data = json.loads(json_str)

        print(f"Extracted CV data: {cv_data}")

        return {
            "cv_text": cv_text,
            "cv_data": cv_data,
            "waiting_for_cv": False
        }
    except Exception as e:
        print(f"Error parsing CV: {e}")
        # Create minimal CV data on error
        return {
            "cv_text": cv_text,
            "cv_data": {
                "skills": [],
                "experience_years": 0,
                "education": "Unknown",
                "job_titles": [],
                "industries": []
            },
            "waiting_for_cv": False
        }

def rank_jobs_with_llm(state: JobScraperState) -> Dict[str, Any]:
    """Rank jobs by relevance to the user's CV using LLM."""
    filtered_jobs = state.get("filtered_jobs", [])
    cv_data = state.get("cv_data", {})
    cv_text = state.get("cv_text", "")

    print(f"Ranking {len(filtered_jobs)} jobs using LLM...")

    if not cv_text:
        print("No CV provided, returning unranked jobs")
        return {"ranked_jobs": filtered_jobs}

    ranked_jobs = []

    # Process in batches to avoid token limits
    batch_size = 3

    for i in range(0, len(filtered_jobs), batch_size):
        batch = filtered_jobs[i:i+batch_size]

        # Create batch-specific prompt
        jobs_text = ""
        for idx, job in enumerate(batch):
            jobs_text += f"\nJob {idx+1}:\n"
            jobs_text += f"Title: {job.get('job_title', 'Unknown')}\n"
            jobs_text += f"Company: {job.get('company_name', 'Unknown')}\n"
            jobs_text += f"Description: {job.get('description', 'No description')[:500]}...\n"
            jobs_text += f"Experience Required: {job.get('experience_years', 0)} years\n"
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
            """
            ),
            HumanMessage(content=f"Here are the jobs to evaluate: {jobs_text}")
        ])

        try:
            response = llm.invoke(prompt)

            # Extract JSON from response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content

            # Try to clean up any non-JSON content
            json_str = re.sub(r'^[^[]*', '', json_str)
            json_str = re.sub(r'[^]]*$', '', json_str)

            rankings = json.loads(json_str)

            # Add relevance scores to jobs
            for ranking in rankings:
                job_idx = ranking.get("job_index", 0) - 1
                if 0 <= job_idx < len(batch):
                    job_copy = batch[job_idx].copy()
                    job_copy["relevance_score"] = ranking.get("score", 0)
                    job_copy["relevance_explanation"] = ranking.get("explanation", "")
                    ranked_jobs.append(job_copy)

        except Exception as e:
            print(f"Error ranking batch: {e}")
            # On error, add jobs without ranking
            for job in batch:
                job_copy = job.copy()
                job_copy["relevance_score"] = 0
                job_copy["relevance_explanation"] = "Error in ranking"
                ranked_jobs.append(job_copy)

    # Sort by relevance score (highest first)
    ranked_jobs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    print(f"Ranked {len(ranked_jobs)} jobs")

    return {
        "ranked_jobs": ranked_jobs,
        "waiting_for_job_selection": True
    }

def prepare_job_selection(state: JobScraperState) -> Dict[str, Any]:
    """Prepare the job selection prompt for the user."""
    ranked_jobs = state.get("ranked_jobs", [])

    if not ranked_jobs:
        return {"waiting_for_job_selection": False}

    # This function would normally prepare a prompt for the user
    # For now, we just set the flag to wait for user input
    return {"waiting_for_job_selection": True}

def collect_job_selection(state: JobScraperState) -> Dict[str, Any]:
    """Collect user's job selection for CV optimization."""
    if not state.get("waiting_for_job_selection"):
        return {}

    user_input = state.get("user_input", "")
    if not user_input:
        return {}

    ranked_jobs = state.get("ranked_jobs", [])
    if not ranked_jobs:
        return {"selected_jobs": []}

    print(f"Processing job selection: {user_input}")

    try:
        # Use LLM to parse the selection
        job_titles = [f"{i+1}. {job.get('job_title', '')} at {job.get('company_name', '')}"
                     for i, job in enumerate(ranked_jobs[:10])]
        job_titles_str = "\n".join(job_titles)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
            Parse the user's job selection from the following list:
            {job_titles_str}
            
            The user may specify jobs by number, title, or company.
            Return a JSON array containing only the job numbers (1-based indexing).
            For example: [1, 3, 5]
            """
            ),
            HumanMessage(content=user_input)
        ])

        response = llm.invoke(prompt)

        # Extract JSON from response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content

        # Try to clean up any non-JSON content
        json_str = re.sub(r'^[^[]*', '', json_str)
        json_str = re.sub(r'[^]]*$', '', json_str)

        selections = json.loads(json_str)

        # Get URLs for selected jobs
        selected_jobs = []
        for idx in selections:
            if isinstance(idx, int) and 1 <= idx <= len(ranked_jobs):
                job = ranked_jobs[idx-1]
                selected_jobs.append(job.get("url", ""))

        if not selected_jobs and len(ranked_jobs) > 0:
            # Fallback: select top job if parsing failed
            selected_jobs = [ranked_jobs[0].get("url", "")]

        print(f"Selected {len(selected_jobs)} jobs: {selected_jobs}")

        return {
            "selected_jobs": selected_jobs,
            "waiting_for_job_selection": False
        }
    except Exception as e:
        print(f"Error parsing job selection: {e}")
        # Fallback to top job
        if ranked_jobs:
            return {
                "selected_jobs": [ranked_jobs[0].get("url", "")],
                "waiting_for_job_selection": False
            }
        return {
            "selected_jobs": [],
            "waiting_for_job_selection": False
        }

def optimize_cvs(state: JobScraperState) -> Dict[str, Any]:
    """Optimize the user's CV for each selected job."""
    selected_jobs = state.get("selected_jobs", [])
    cv_text = state.get("cv_text", "")
    cv_data = state.get("cv_data", {})
    ranked_jobs = state.get("ranked_jobs", [])

    if not selected_jobs or not cv_text:
        return {"optimized_cvs": {}}

    print(f"Optimizing CV for {len(selected_jobs)} jobs...")

    optimized_cvs = {}

    # Find the full job details for each selected URL
    for job_url in selected_jobs:
        job_details = None
        for job in ranked_jobs:
            if job.get("url") == job_url:
                job_details = job
                break

        if not job_details:
            continue

        job_title = job_details.get("job_title", "Unknown Position")
        company_name = job_details.get("company_name", "Unknown Company")
        job_description = job_details.get("description", "")

        try:
            # Use LLM to optimize the CV
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
                """
                ),
                HumanMessage(content=f"Original CV:\n\n{cv_text}")
            ])

            response = llm.invoke(prompt)
            optimized_cv = response.content

            print(f"Optimized CV for {job_title} at {company_name}")

            # Store with a descriptive key
            key = f"{company_name} - {job_title}"
            optimized_cvs[key] = optimized_cv

        except Exception as e:
            print(f"Error optimizing CV for {job_title}: {e}")

    return {"optimized_cvs": optimized_cvs}

def initial_state_creator() -> JobScraperState:
    """Create the initial state for the graph."""
    return {
        "job_title": "",
        "max_jobs": 20,
        "user_input": None,
        "job_urls": None,
        "jobs_data": None,
        "user_preferences": None,
        "available_locations": None,
        "waiting_for_preferences": None,
        "filtered_jobs": None,
        "ranked_jobs": None,
        "cv_text": None,
        "cv_data": None,
        "waiting_for_cv": None,
        "waiting_for_job_selection": None,
        "selected_jobs": None,
        "optimized_cvs": None,
        "error": None
    }

def should_wait_for_preferences(state: JobScraperState) -> str:
    """Determine if we should wait for user preferences."""
    if state.get("waiting_for_preferences", False):
        if state.get("user_input"):
            return "collect_user_preferences"
        return "wait_for_input"
    return "filter_jobs"

def should_wait_for_cv(state: JobScraperState) -> str:
    """Determine if we should wait for CV."""
    if state.get("waiting_for_cv", False):
        if state.get("user_input"):
            return "collect_cv"
        return "wait_for_input"
    return "rank_jobs"

def should_wait_for_job_selection(state: JobScraperState) -> str:
    """Determine if we should wait for job selection."""
    if state.get("waiting_for_job_selection", False):
        if state.get("user_input"):
            return "collect_job_selection"
        return "wait_for_input"
    return "optimize_cvs"

# Build the graph
def build_graph():
    """Build and return the job application assistant graph."""
    graph = StateGraph(JobScraperState)

    # Add nodes
    graph.add_node("search_jobs", search_jobs_node)
    graph.add_node("prepare_user_preferences", prepare_user_preference_prompt)
    graph.add_node("collect_user_preferences", collect_user_preferences)
    graph.add_node("filter_jobs", filter_jobs_node)
    graph.add_node("collect_cv", collect_cv)
    graph.add_node("rank_jobs", rank_jobs_with_llm)
    graph.add_node("prepare_job_selection", prepare_job_selection)
    graph.add_node("collect_job_selection", collect_job_selection)
    graph.add_node("optimize_cvs", optimize_cvs)
    graph.add_node("wait_for_input", lambda x: x)  # No-op node for waiting

    # Add edges
    graph.set_entry_point("search_jobs")
    graph.add_edge("search_jobs", "prepare_user_preferences")

    # Conditional edges for user preference collection
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

    # Conditional edges for CV collection
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

    # Job selection
    graph.add_edge("rank_jobs", "prepare_job_selection")

    # Conditional edges for job selection
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

    # Final edge
    graph.add_edge("optimize_cvs", END)

    return graph.compile()

# Create the graph
graph = build_graph()
