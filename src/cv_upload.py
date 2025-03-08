"""
CV file upload handler for the Job Scraper Agent.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import os

from src.tools.file_utils import extract_text_from_file
from src.graph import graph, initial_state_creator, JobScraperState

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def process_cv_file(
        file_path: Optional[str] = None,
        file_content: Optional[bytes] = None,
        file_name: Optional[str] = None,
        current_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a CV file and prepare it for the graph.

    This function can be used two ways:
    1. To extract text from a file and create/update a state that's ready for the graph
    2. To directly invoke the graph with the extracted text

    Args:
        file_path: Path to the file (optional)
        file_content: Binary content of the file (optional)
        file_name: Name of the file (optional)
        current_state: Current graph state (optional)

    Returns:
        Updated state or graph result
    """
    logger.info(f"Processing CV file: {file_name or file_path}")

    # Extract text from the file
    result = extract_text_from_file(file_path, file_content, file_name)

    if not result['success']:
        logger.error(f"Error extracting text from file: {result['error']}")
        return {'error': result['error']}

    logger.info(
        f"Successfully extracted {len(result['text'])} characters from file")

    # Get or create graph state
    state = current_state.copy() if current_state else initial_state_creator()

    if file_path:
        # Store the file path in the state
        state['cv_file_path'] = file_path
    if file_name:
        # Store the file name in the state
        state['cv_file_name'] = file_name

    # Set the extracted text as user input
    # This maintains compatibility with the existing flow
    state['user_input'] = result['text']

    return state


async def process_cv_file_and_invoke_graph(
        file_path: Optional[str] = None,
        file_content: Optional[bytes] = None,
        file_name: Optional[str] = None,
        current_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a CV file and invoke the graph with the extracted text.

    Args:
        file_path: Path to the file (optional)
        file_content: Binary content of the file (optional)
        file_name: Name of the file (optional)
        current_state: Current graph state (optional)

    Returns:
        Graph result
    """
    # Process the file
    state = await process_cv_file(file_path, file_content, file_name,
                                  current_state)

    if 'error' in state:
        return state

    # Make sure we're in the right state to process a CV
    state['waiting_for_cv'] = True

    # Invoke the graph
    logger.info("Invoking graph with extracted CV text")
    config = {"configurable": {"thread_id": "file-upload-thread"}}
    try:
        result = await graph.ainvoke(state, config=config)
        logger.info("Graph invocation successful")
        return result
    except Exception as e:
        logger.error(f"Error invoking graph: {str(e)}")
        return {'error': str(e)}