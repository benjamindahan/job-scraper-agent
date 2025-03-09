"""
Scraping tools for job listings with enhanced anti-blocking measures.
"""

import re
import asyncio
import aiohttp
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import logging
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base URL for the job site
BASE_URL = "https://jobs.secrettelaviv.com"

# Database path (used for caching)
DB_PATH = "jobs.db"

# Rotating user agents to avoid blocking
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
]

def get_random_user_agent():
    """Get a random user agent from the list."""
    return random.choice(USER_AGENTS)

async def fetch_page(url: str, session: aiohttp.ClientSession, retry_count: int = 3) -> Optional[str]:
    """
    Fetch a page using aiohttp with exponential backoff retry logic and rotating user agents.

    Args:
        url: URL to fetch
        session: aiohttp ClientSession
        retry_count: Number of retries on failure

    Returns:
        HTML content as string or None if failed
    """
    base_delay = 0.5  # Start with 500ms delay

    for attempt in range(retry_count + 1):
        try:
            # Add delay with exponential backoff
            if attempt > 0:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
                logger.info(f"Retry attempt {attempt}/{retry_count} for {url} after {delay:.2f}s delay")
            else:
                # Initial delay
                await asyncio.sleep(base_delay)

            # Get a random user agent for each request
            user_agent = get_random_user_agent()

            async with session.get(url, headers={
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }) as response:
                if response.status == 200:
                    return await response.text()

                if response.status == 429:  # Too Many Requests
                    if attempt < retry_count:
                        logger.warning(f"Rate limited while fetching {url}. Will retry in {base_delay * (2 ** (attempt+1)):.2f} seconds.")
                        continue
                    else:
                        logger.error(f"Failed to fetch {url} after {retry_count} retries: HTTP 429")
                        return None
                else:
                    logger.error(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            if attempt < retry_count:
                continue
            return None

    return None


def check_cached_jobs(job_title: str, max_age_hours: int = 24) -> List[Dict[str, Any]]:
    """
    Check for cached job listings in the database.

    Args:
        job_title: The job title to search for
        max_age_hours: Maximum age of cached results in hours

    Returns:
        List of cached job dictionaries or empty list if no valid cache
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Calculate the cutoff time
        current_time = datetime.now()
        cutoff_time = (current_time - timedelta(hours=max_age_hours)).isoformat()

        # Look for jobs matching the title and within the age limit
        query = """
        SELECT * FROM jobs 
        WHERE job_title LIKE ? AND created_at > ?
        ORDER BY publication_date DESC
        """

        cursor.execute(query, (f"%{job_title}%", cutoff_time))

        jobs = []
        for row in cursor.fetchall():
            job = dict(row)
            if 'publication_date' in job and isinstance(job['publication_date'], str):
                try:
                    job['publication_date'] = datetime.fromisoformat(job['publication_date'])
                except ValueError:
                    pass
            jobs.append(job)

        conn.close()

        if jobs:
            logger.info(f"Found {len(jobs)} cached jobs for '{job_title}' from the last {max_age_hours} hours")

        return jobs
    except Exception as e:
        logger.error(f"Error checking cached jobs: {str(e)}")
        return []


async def search_jobs(query: str, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    """
    Search for jobs with the given query and return basic information and URLs.
    First checks cached results, falls back to scraping if needed.
    """
    # Check for cached results first
    cached_jobs = check_cached_jobs(query)
    if cached_jobs:
        # If we have cached results, use them instead of scraping
        logger.info(f"Using {len(cached_jobs)} cached job listings for query: {query}")
        return cached_jobs

    # If no cached results, proceed with scraping
    search_url = f"{BASE_URL}/list/find/?query={'+'.join(query.split())}"
    html_content = await fetch_page(search_url, session, retry_count=5)  # Increased retries

    if not html_content:
        logger.warning(f"No jobs found for query: {query}")
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    job_listings = []

    for job_card in soup.select('.wpjb-job-list .wpjb-grid-row'):
        try:
            # Extract job title and URL
            title_elem = job_card.select_one('.wpjb-col-title .wpjb-line-major a')
            if not title_elem:
                continue

            job_title = title_elem.text.strip()
            job_url = title_elem.get('href')

            # Extract company name
            company_elem = job_card.select_one('.wpjb-col-title .wpjb-sub-small')
            company_name = company_elem.text.strip() if company_elem else "Unknown Company"

            # Extract location
            location_elem = job_card.select_one('.wpjb-col-location .wpjb-line-major')
            location = location_elem.text.strip() if location_elem else "Unknown Location"

            # Extract job type
            job_type_elem = job_card.select_one('.wpjb-col-location .wpjb-sub-small')
            job_type = job_type_elem.text.strip() if job_type_elem else "Unknown Type"

            # Extract date
            date_elem = job_card.select_one('.wpjb-col-15 .wpjb-line-major')
            date_posted = date_elem.text.strip() if date_elem else "Unknown Date"

            # Check if job is new
            is_new = bool(job_card.select_one('.wpjb-bulb'))

            # Add job to listings
            job_listings.append({
                'job_title': job_title,
                'company_name': company_name,
                'location': location,
                'job_type': job_type,
                'date_preview': date_posted,
                'is_new': is_new,
                'job_url': job_url if job_url.startswith('http') else f"{BASE_URL}{job_url}",
            })
        except Exception as e:
            logger.error(f"Error parsing job card: {str(e)}")

    return job_listings


def parse_experience_years(experience_text: str) -> int:
    """Extract the numeric value for years of experience from text."""
    if not experience_text:
        return 0

    more_than_match = re.search(r"More than (\d+)", experience_text)
    if more_than_match:
        return int(more_than_match.group(1))

    years_match = re.search(r"(\d+)", experience_text)
    if years_match:
        return int(years_match.group(1))

    return 0


def parse_publication_date(date_text: str) -> datetime:
    """Convert publication date text to datetime object."""
    if not date_text:
        return datetime.now()

    match = re.search(r"Published: (.+)", date_text)
    if match:
        date_str = match.group(1).strip()
        try:
            return datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            logger.error(f"Could not parse date string: {date_str}")

    return datetime.now()


async def extract_job_details(job_url: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
    """
    Extract detailed information from a job listing page.
    """
    html_content = await fetch_page(job_url, session, retry_count=3)
    if not html_content:
        return {}

    soup = BeautifulSoup(html_content, 'html.parser')

    try:
        # Extract job title
        job_title_elem = soup.select_one('.post-title')
        job_title = job_title_elem.text.strip() if job_title_elem else "Unknown Title"

        # Extract company name - FIXED to get only the company name
        company_elem = soup.select_one('.wpjb-top-header-title')
        company_name = "Unknown Company"
        if company_elem:
            # Use regex to extract only alphabetic characters for the company name
            company_text = company_elem.get_text(strip=True)
            # Match only alphabetic characters until we hit a digit or special character
            match = re.match(r'^([A-Za-z\s\-\.&]+)', company_text)
            if match:
                company_name = match.group(1).strip()
            else:
                # Fallback: take the first word if regex fails
                company_name = company_text.split()[0]

        # Extract publication date
        pub_date_elem = soup.select_one('.wpjb-top-header-published')
        pub_date_text = pub_date_elem.text.strip() if pub_date_elem else ""
        publication_date = parse_publication_date(pub_date_text)

        # Extract category
        category_elem = soup.select_one('.wpjb-row-meta-_tag_category .wpjb-grid-col.wpjb-col-60')
        category = category_elem.text.strip() if category_elem else "Unknown Category"

        # Extract job type - FIXED to clean up extra text
        job_type_elem = soup.select_one('.wpjb-row-meta-_tag_type .wpjb-grid-col.wpjb-col-60')
        job_type = "Unknown Type"
        if job_type_elem:
            job_type_text = job_type_elem.text.strip()
            # Extract only the job type from text like "Full-time" without extra numbers
            match = re.search(r'(Full-time|Part-time|Contract|Temporary|Internship)', job_type_text)
            if match:
                job_type = match.group(1)
            else:
                job_type = job_type_text

        # Extract education level
        education_elem = soup.select_one('.wpjb-row-meta-education .wpjb-grid-col.wpjb-col-60')
        education_level = education_elem.text.strip() if education_elem else "Unknown Education"

        # Extract location
        location_elem = soup.select_one('.wpjb-row-meta-location_stlv .wpjb-grid-col.wpjb-col-60')
        location = location_elem.text.strip() if location_elem else "Unknown Location"

        # Extract experience level
        experience_elem = soup.select_one('.wpjb-row-meta-years .wpjb-grid-col.wpjb-col-60')
        experience_text = experience_elem.text.strip() if experience_elem else ""
        experience_years = parse_experience_years(experience_text)

        # Extract job description
        description_elem = soup.select_one('.wpjb-text-box .wpjb-text')
        description = description_elem.get_text("\n", strip=True) if description_elem else "No description available"

        # Return structured job data
        return {
            'job_title': job_title,
            'company_name': company_name,
            'publication_date': publication_date,
            'category': category,
            'job_type': job_type,
            'education_level': education_level,
            'location': location,
            'experience_years': experience_years,
            'experience_text': experience_text,
            'description': description,
            'url': job_url,
        }
    except Exception as e:
        logger.error(f"Error extracting job details from {job_url}: {str(e)}")
        return {}


async def scrape_jobs(job_title: str, max_jobs: int = 20) -> List[
    Dict[str, Any]]:
    """
    Main function to search for jobs and scrape detailed information.
    Will use cached results if available.

    Args:
        job_title: The job title to search for
        max_jobs: Maximum number of jobs to retrieve

    Returns:
        List of job dictionaries with detailed information
    """
    # First check for cached results
    cached_jobs = check_cached_jobs(job_title)
    if cached_jobs:
        # If we have enough cached jobs, return them directly
        if len(cached_jobs) >= max_jobs:
            logger.info(
                f"Using {min(len(cached_jobs), max_jobs)} cached jobs for query: {job_title}")

            # Add the original search query to each job
            for job in cached_jobs:
                job['searched_job_title'] = job_title.lower().strip()

            return cached_jobs[:max_jobs]
        # Otherwise, we'll still do scraping but log that we're supplementing
        logger.info(
            f"Found {len(cached_jobs)} cached jobs, but need {max_jobs}. Will scrape additional jobs.")

    async with aiohttp.ClientSession() as session:
        # Search for jobs
        job_listings = await search_jobs(job_title, session)

        if not job_listings:
            # If scraping failed but we have some cached results, return those
            if cached_jobs:
                logger.warning(
                    f"Scraping failed, using {len(cached_jobs)} cached jobs for query: {job_title}")

                # Add the original search query to each job
                for job in cached_jobs:
                    job['searched_job_title'] = job_title.lower().strip()

                return cached_jobs
            logger.warning(f"No jobs found for query: {job_title}")
            return []

        # Limit the number of jobs to scrape
        job_listings = job_listings[:max_jobs]

        # Add random pauses between job detail requests to avoid detection
        tasks = []
        for i, job in enumerate(job_listings):
            if i > 0:
                # Random pause between detail requests (0.5 to 2.5 seconds)
                pause = 0.5 + random.random() * 2.0
                await asyncio.sleep(pause)
            tasks.append(extract_job_details(job['job_url'], session))

        job_details = await asyncio.gather(*tasks)

        # Filter out empty results
        jobs = [job for job in job_details if job]

        # Add the original search query to each job
        for job in jobs:
            job['searched_job_title'] = job_title.lower().strip()

        if not jobs and cached_jobs:
            # If all scraping failed but we have cached results, use those
            logger.warning(
                f"Detailed scraping failed, using {len(cached_jobs)} cached jobs")

            # Add the original search query to each cached job
            for job in cached_jobs:
                job['searched_job_title'] = job_title.lower().strip()

            return cached_jobs

        return jobs


# Example usage (for testing)
if __name__ == "__main__":
    import sys

    async def main():
        query = "data scientist" if len(sys.argv) < 2 else sys.argv[1]
        max_jobs = 5 if len(sys.argv) < 3 else int(sys.argv[2])

        print(f"Searching for '{query}' jobs (max: {max_jobs})...")
        jobs = await scrape_jobs(query, max_jobs)

        print(f"Found {len(jobs)} jobs:")
        for i, job in enumerate(jobs, 1):
            print(f"\n{i}. {job['job_title']} at {job['company_name']}")
            print(f"   Location: {job['location']}")
            print(f"   Experience: {job['experience_years']} years ({job['experience_text']})")
            print(f"   Published: {job['publication_date'].strftime('%Y-%m-%d')}")
            print(f"   URL: {job['url']}")
            print(f"   Description preview: {job['description'][:100]}...")

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())