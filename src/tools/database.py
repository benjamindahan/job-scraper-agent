"""
Database utilities for storing and retrieving job listings.
"""

import sqlite3
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = "jobs.db"


def init_database() -> None:
    """
    Initialize the database with the necessary tables.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create jobs table with new searched_job_title column
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_title TEXT NOT NULL,
        company_name TEXT NOT NULL,
        publication_date TEXT NOT NULL,
        category TEXT,
        job_type TEXT,
        education_level TEXT,
        location TEXT,
        experience_years INTEGER,
        experience_text TEXT,
        description TEXT,
        url TEXT UNIQUE,
        created_at TEXT NOT NULL,
        searched_job_title TEXT
    )
    ''')

    # Check if searched_job_title column exists, add it if not
    cursor.execute("PRAGMA table_info(jobs)")
    columns = [column[1] for column in cursor.fetchall()]

    if 'searched_job_title' not in columns:
        logger.info("Adding searched_job_title column to jobs table")
        cursor.execute('ALTER TABLE jobs ADD COLUMN searched_job_title TEXT')

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def store_jobs(jobs: List[Dict[str, Any]], search_query: Optional[str] = None) -> int:
    """
    Store multiple job listings in the database.

    Args:
        jobs: List of job dictionaries to store
        search_query: The original search query used to find these jobs

    Returns:
        Number of jobs successfully stored
    """
    if not jobs:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    count = 0
    for job in jobs:
        try:
            # Convert datetime to ISO format string
            if isinstance(job.get('publication_date'), datetime):
                job['publication_date'] = job['publication_date'].isoformat()

            # Include the search query as searched_job_title
            job_search_query = search_query or job.get('searched_job_title', '')

            cursor.execute('''
            INSERT OR REPLACE INTO jobs 
            (job_title, company_name, publication_date, category, job_type, 
             education_level, location, experience_years, experience_text, description, url, created_at, searched_job_title)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job.get('job_title', ''),
                job.get('company_name', ''),
                job.get('publication_date', ''),
                job.get('category', ''),
                job.get('job_type', ''),
                job.get('education_level', ''),
                job.get('location', ''),
                job.get('experience_years', 0),
                job.get('experience_text', ''),
                job.get('description', ''),
                job.get('url', ''),
                datetime.now().isoformat(),
                job_search_query
            ))
            count += 1
        except Exception as e:
            logger.error(f"Error storing job {job.get('job_title')}: {str(e)}")

    conn.commit()
    conn.close()

    logger.info(f"Stored {count} jobs in the database")
    return count


def get_all_jobs() -> List[Dict[str, Any]]:
    """
    Retrieve all jobs from the database.

    Returns:
        List of job dictionaries
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM jobs ORDER BY publication_date DESC')

    jobs = []
    for row in cursor.fetchall():
        job = dict(row)
        # Convert publication_date string back to datetime if needed
        if 'publication_date' in job and isinstance(job['publication_date'],
                                                    str):
            try:
                job['publication_date'] = datetime.fromisoformat(
                    job['publication_date'])
            except ValueError:
                pass
        jobs.append(job)

    conn.close()
    logger.info(f"Retrieved {len(jobs)} jobs from database")
    return jobs

def get_jobs_by_title(job_title: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get jobs matching a specific title from the database.

    Args:
        job_title: The job title to search for
        limit: Maximum number of jobs to return

    Returns:
        List of job dictionaries
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Improved search using OR and LIKE for searched_job_title
    query = """
    SELECT * FROM jobs WHERE 
    job_title LIKE ? 
    OR searched_job_title LIKE ? 
    ORDER BY publication_date DESC LIMIT ?
    """

    search_term = f"%{job_title}%"
    cursor.execute(query, (search_term, search_term, limit))

    jobs = []
    for row in cursor.fetchall():
        job = dict(row)
        # Convert publication_date string back to datetime if needed
        if 'publication_date' in job and isinstance(job['publication_date'], str):
            try:
                job['publication_date'] = datetime.fromisoformat(
                    job['publication_date'])
            except ValueError:
                pass
        jobs.append(job)

    conn.close()
    logger.info(f"Retrieved {len(jobs)} jobs matching '{job_title}' from database")
    return jobs

def get_unique_locations() -> List[str]:
    """
    Get a list of unique job locations from the database.

    Returns:
        List of unique location strings
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        'SELECT DISTINCT location FROM jobs WHERE location IS NOT NULL AND location != ""')

    locations = [row[0] for row in cursor.fetchall()]
    conn.close()

    return locations


def filter_jobs(min_experience: Optional[int] = None,
                max_experience: Optional[int] = None,
                locations: Optional[List[str]] = None,
                date_range: Optional[str] = None,
                search_query: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Filter jobs based on user preferences with enhanced date handling.

    Args:
        min_experience: Minimum years of experience required
        max_experience: Maximum years of experience (new parameter)
        locations: List of acceptable locations
        date_range: Time filter (e.g., "Last 24 hours", "Last Week", "Last Month", "Last 2 Weeks")
        search_query: Original search query to filter by job title

    Returns:
        List of filtered job dictionaries
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = 'SELECT * FROM jobs WHERE 1=1'
    params = []

    # Add experience filter - with min and max bounds
    if min_experience is not None:
        query += ' AND (experience_years >= ? OR experience_years = 0)'  # Include jobs that don't specify experience
        params.append(min_experience)

    if max_experience is not None:
        query += ' AND (experience_years <= ? OR experience_years = 0)'  # Include jobs that don't specify experience
        params.append(max_experience)

    # Add location filter
    if locations and len(locations) > 0:
        placeholders = ','.join(['?' for _ in locations])
        query += f' AND location IN ({placeholders})'
        params.extend(locations)

    # Add date filter with enhanced handling
    if date_range:
        current_date = datetime.now()
        days = None

        # Handle various date range formats
        if date_range == "Last 24 hours":
            days = 1
        elif date_range == "Last 48 hours":
            days = 2
        elif date_range == "Last Week":
            days = 7
        elif date_range == "Last 2 Weeks":
            days = 14
        elif date_range == "Last Month":
            days = 30
        elif date_range.lower().startswith("last "):
            # Try to parse custom ranges like "Last X days"
            try:
                parts = date_range.lower().split()
                if len(parts) >= 3 and parts[0] == "last" and parts[2] in ["days", "day", "weeks", "week"]:
                    num = int(parts[1])
                    if parts[2] in ["weeks", "week"]:
                        days = num * 7
                    else:
                        days = num
            except (ValueError, IndexError):
                days = None

        if days:
            cutoff_date = (current_date - timedelta(days=days)).isoformat()
            query += ' AND publication_date >= ?'
            params.append(cutoff_date)

    # Add search query filter
    if search_query:
        # Search in both job_title and searched_job_title for better matching
        query += ' AND (job_title LIKE ? OR searched_job_title LIKE ?)'
        search_term = f"%{search_query}%"
        params.append(search_term)
        params.append(search_term)

    # Order by publication date (newest first)
    query += ' ORDER BY publication_date DESC'

    cursor.execute(query, params)

    jobs = []
    for row in cursor.fetchall():
        job = dict(row)
        # Convert publication_date string back to datetime if needed
        if 'publication_date' in job and isinstance(job['publication_date'],
                                                    str):
            try:
                job['publication_date'] = datetime.fromisoformat(
                    job['publication_date'])
            except ValueError:
                pass
        jobs.append(job)

    conn.close()
    logger.info(f"Filtered to {len(jobs)} jobs matching preferences")
    return jobs


# Initialize the database when the module is imported
init_database()