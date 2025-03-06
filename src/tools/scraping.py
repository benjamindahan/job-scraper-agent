"""
Scraping tools for job listings from jobs.secrettelaviv.com.
"""

import re
import asyncio
import aiohttp
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base URL for the job site
BASE_URL = "https://jobs.secrettelaviv.com"


async def fetch_page(url: str, session: aiohttp.ClientSession) -> Optional[str]:
    """Fetch a page using aiohttp and return the HTML content."""
    try:
        async with session.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }) as response:
            if response.status == 200:
                return await response.text()
            logger.error(f"Failed to fetch {url}: HTTP {response.status}")
            return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


async def search_jobs(query: str, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    """
    Search for jobs with the given query and return basic information and URLs.
    """
    search_url = f"{BASE_URL}/list/find/?query={'+'.join(query.split())}"
    html_content = await fetch_page(search_url, session)

    if not html_content:
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
    html_content = await fetch_page(job_url, session)
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


async def scrape_jobs(job_title: str, max_jobs: int = 20) -> List[Dict[str, Any]]:
    """
    Main function to search for jobs and scrape detailed information.
    """
    async with aiohttp.ClientSession() as session:
        # Search for jobs
        job_listings = await search_jobs(job_title, session)

        if not job_listings:
            logger.warning(f"No jobs found for query: {job_title}")
            return []

        # Limit the number of jobs to scrape
        job_listings = job_listings[:max_jobs]

        # Scrape detailed information for each job
        tasks = [extract_job_details(job['job_url'], session) for job in job_listings]
        job_details = await asyncio.gather(*tasks)

        # Filter out empty results
        return [job for job in job_details if job]


# Example usage (for testing)
if __name__ == "__main__":
    import sys

    async def main():
        query = "data scientist" if len(sys.argv) < 2 else sys.argv[1]
        max_jobs = 5 if len(sys.argv) < 3 else int(sys.argv[2])

        print(f"Searching for '{query}' jobs (max: {max_jobs})...")
        jobs = await scrape_jobs(query, max_jobs)

        print(jobs[0])

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