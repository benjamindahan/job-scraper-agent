"""
Database utilities for storing and retrieving job listings.
"""

import sqlite3
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
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

    # Create jobs table
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
        created_at TEXT NOT NULL
    )
    ''')

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def store_jobs(jobs: List[Dict[str, Any]]) -> int:
    """
    Store multiple job listings in the database.

    Args:
        jobs: List of job dictionaries to store

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

            cursor.execute('''
            INSERT OR REPLACE INTO jobs 
            (job_title, company_name, publication_date, category, job_type, 
             education_level, location, experience_years, experience_text, description, url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                datetime.now().isoformat()
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
                locations: Optional[List[str]] = None,
                date_range: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Filter jobs based on user preferences.

    Args:
        min_experience: Minimum years of experience required
        locations: List of acceptable locations
        date_range: Time filter (e.g., "Last 24 hours", "Last Week", "Last Month")

    Returns:
        List of filtered job dictionaries
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = 'SELECT * FROM jobs WHERE 1=1'
    params = []

    # Add experience filter
    if min_experience is not None:
        query += ' AND experience_years >= ?'
        params.append(min_experience)

    # Add location filter
    if locations and len(locations) > 0:
        placeholders = ','.join(['?' for _ in locations])
        query += f' AND location IN ({placeholders})'
        params.extend(locations)

    # Add date filter
    if date_range:
        current_date = datetime.now()

        if date_range == "Last 24 hours":
            days = 1
        elif date_range == "Last Week":
            days = 7
        elif date_range == "Last Month":
            days = 30
        else:
            days = None

        if days:
            cutoff_date = (current_date - datetime.timedelta(
                days=days)).isoformat()
            query += ' AND publication_date >= ?'
            params.append(cutoff_date)

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

    return jobs


# Initialize the database when the module is imported
init_database()