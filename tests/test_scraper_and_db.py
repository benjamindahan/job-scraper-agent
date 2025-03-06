"""
Test script for scraper and database integration.
"""

import asyncio
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.scraping import scrape_jobs
from src.tools.database import store_jobs, get_all_jobs, get_unique_locations, \
    filter_jobs


async def test_scrape_and_store():
    # Step 1: Scrape jobs
    print("Scraping jobs...")
    job_title = "data scientist"
    max_jobs = 10
    jobs = await scrape_jobs(job_title, max_jobs)
    print(f"Scraped {len(jobs)} jobs")

    # Print the first job for inspection
    if jobs:
        print("\nSample job details:")
        job = jobs[0]
        for key, value in job.items():
            print(f"  {key}: {value}")

    # Step 2: Store jobs in database
    print("\nStoring jobs in database...")
    stored_count = store_jobs(jobs)
    print(f"Stored {stored_count} jobs")

    # Step 3: Retrieve all jobs
    print("\nRetrieving all jobs from database...")
    all_jobs = get_all_jobs()
    print(f"Retrieved {len(all_jobs)} jobs")

    # Step 4: Get unique locations
    print("\nUnique job locations:")
    locations = get_unique_locations()
    for location in locations:
        print(f"  - {location}")

    # Step 5: Test filtering
    print("\nFiltering jobs with 3+ years experience:")
    filtered_jobs = filter_jobs(min_experience=3)
    print(f"Found {len(filtered_jobs)} matching jobs")

    if filtered_jobs:
        print("\nFiltered job titles:")
        for job in filtered_jobs[:5]:  # Show up to 5 jobs
            print(
                f"  - {job['job_title']} at {job['company_name']} ({job['experience_years']} years)")

    return True


if __name__ == "__main__":
    # For Windows compatibility
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(test_scrape_and_store())
    print("\nTest completed successfully!")