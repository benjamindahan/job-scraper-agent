import streamlit as st
import asyncio
import os
import sys
from tempfile import NamedTemporaryFile
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.graph import graph, initial_state_creator
from src.tools.database import get_unique_locations

# Set page config
st.set_page_config(
    page_title="Job Scraper Agent",
    page_icon="💼",
    layout="wide"
)

# Configure asyncio for Windows compatibility
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Main UI
st.title("💼 Job Scraper Agent")
st.markdown("Find the best jobs matching your CV and preferences.")

# Session state initialization
if 'state' not in st.session_state:
    st.session_state.state = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'job_search_complete' not in st.session_state:
    st.session_state.job_search_complete = False
if 'preferences_complete' not in st.session_state:
    st.session_state.preferences_complete = False
if 'cv_complete' not in st.session_state:
    st.session_state.cv_complete = False
if 'job_selection_complete' not in st.session_state:
    st.session_state.job_selection_complete = False

# Create tabs for workflow
tab1, tab2, tab3, tab4 = st.tabs(
    ["Job Search", "Preferences", "CV Analysis", "Results"])

with tab1:
    st.header("Step 1: Search for Jobs")

    col1, col2 = st.columns([3, 1])

    with col1:
        job_title = st.text_input("Job Title",
                                  placeholder="e.g. Data Scientist")

    with col2:
        max_jobs = st.number_input("Max Jobs", min_value=5, max_value=50,
                                   value=20)

    use_db_only = st.checkbox("Use database only (faster, no new scraping)")

    if st.button("Search Jobs", type="primary",
                 disabled=st.session_state.job_search_complete):
        if not job_title:
            st.error("Please enter a job title")
        else:
            with st.spinner("Searching for jobs..."):
                # Initialize the state
                state = initial_state_creator()
                state["job_title"] = job_title
                state["max_jobs"] = max_jobs

                # Create a unique session ID
                session_id = f"streamlit-session-{int(time.time())}"
                config = {"configurable": {"thread_id": session_id}}

                # Run the graph to get jobs
                if use_db_only:
                    from src.tools.database import get_jobs_by_title, \
                        get_all_jobs

                    db_jobs = get_jobs_by_title(job_title, max_jobs)

                    if not db_jobs:
                        st.warning(
                            "No jobs found in database. Checking all available jobs...")
                        db_jobs = get_all_jobs()

                    if db_jobs:
                        state["jobs_data"] = db_jobs
                        state["job_urls"] = [job.get("url", "") for job in
                                             db_jobs]
                        st.success(f"Found {len(db_jobs)} jobs in database")
                    else:
                        st.error("No jobs found in database")
                else:
                    try:
                        # Run the async code
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        state = loop.run_until_complete(
                            graph.ainvoke(state, config=config))

                        if state.get("error"):
                            st.error(f"Error: {state['error']}")
                            # Fall back to database
                            from src.tools.database import get_jobs_by_title, \
                                get_all_jobs

                            db_jobs = get_jobs_by_title(job_title, max_jobs)
                            if db_jobs:
                                state["jobs_data"] = db_jobs
                                state["job_urls"] = [job.get("url", "") for job
                                                     in db_jobs]
                                st.success(
                                    f"Using {len(db_jobs)} jobs from database instead")
                            else:
                                st.error("No jobs found in database either")
                        else:
                            st.success(
                                f"Found {len(state.get('jobs_data', []))} jobs")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.error("Please try again or use database-only mode")

                # Save state and allow moving to next tab
                if state.get("jobs_data"):
                    st.session_state.state = state
                    st.session_state.job_search_complete = True
                    st.balloons()
                    st.info(
                        "✓ Job search complete! Please proceed to the Preferences tab.")

with tab2:
    st.header("Step 2: Set Your Preferences")

    # Check if we have job data from previous step
    if not st.session_state.job_search_complete:
        st.warning("Please complete the job search step first")
    else:
        state = st.session_state.state

        # Get available locations from the state or directly
        if not state.get("available_locations"):
            locations = get_unique_locations()
            state["available_locations"] = locations
        else:
            locations = state.get("available_locations")

        # Display available locations
        if locations:
            st.subheader("Available Locations")
            location_cols = st.columns(3)
            for i, loc in enumerate(locations[:15]):
                with location_cols[i % 3]:
                    st.write(f"• {loc}")
            if len(locations) > 15:
                st.caption(f"... and {len(locations) - 15} more locations")

        # Preferences form
        st.subheader("Enter Your Preferences")

        col1, col2 = st.columns(2)

        with col1:
            experience = st.text_input("Experience Level",
                                       placeholder="e.g. '3 years', 'entry level', '2-5 years'")

        with col2:
            date_range = st.selectbox("Date Range",
                                      ["Any time", "Last 24 hours",
                                       "Last Week", "Last 2 Weeks",
                                       "Last Month"])

        preferred_locations = st.multiselect("Preferred Locations",
                                             options=locations if locations else [])

        # Combine preferences into a single string
        preferences = f"{experience} experience"
        if preferred_locations:
            preferences += f" in {', '.join(preferred_locations)}"
        preferences += f", posted in {date_range.lower()}"

        st.markdown(f"**Your preferences:** {preferences}")

        if st.button("Apply Preferences", type="primary",
                     disabled=st.session_state.preferences_complete):
            with st.spinner("Filtering jobs based on your preferences..."):
                # Update the state with preferences
                state["preference_input"] = preferences
                state["waiting_for_preferences"] = True
                state["force_refilter"] = True  # Force refiltering

                # Create a unique session ID
                session_id = f"streamlit-session-{int(time.time())}"
                config = {"configurable": {"thread_id": session_id}}

                try:
                    # Run the async code
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    state = loop.run_until_complete(
                        graph.ainvoke(state, config=config))

                    filtered_jobs = state.get("filtered_jobs", [])
                    if not filtered_jobs:
                        st.warning(
                            "No jobs match your exact criteria. Showing all available jobs instead.")
                        filtered_jobs = state.get("jobs_data", [])
                        state["filtered_jobs"] = filtered_jobs

                    st.success(
                        f"Found {len(filtered_jobs)} jobs matching your preferences")

                    # Save updated state
                    st.session_state.state = state
                    st.session_state.preferences_complete = True
                    st.info(
                        "✓ Preferences applied! Please proceed to the CV Analysis tab.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error("Please try again with different preferences")

with tab3:
    st.header("Step 3: CV Analysis & Job Ranking")

    # Check if previous steps are complete
    if not st.session_state.preferences_complete:
        st.warning("Please complete the preferences step first")
    else:
        state = st.session_state.state

        # CV upload or text input
        st.subheader("Upload or Enter Your CV")

        cv_method = st.radio("CV Input Method", ["Upload File", "Enter Text"])

        if cv_method == "Upload File":
            uploaded_file = st.file_uploader("Upload CV",
                                             type=["pdf", "docx", "txt"])

            if uploaded_file:
                # Save the uploaded file to a temporary file
                with NamedTemporaryFile(delete=False,
                                        suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                st.success(f"CV uploaded: {uploaded_file.name}")

                # Save the file path in the state
                state["cv_file_path"] = temp_path
                state["cv_file_name"] = uploaded_file.name
                state["waiting_for_cv"] = True
        else:
            cv_text = st.text_area("Enter your CV text", height=300)
            if cv_text:
                # Save the CV text in the state
                state["user_input"] = cv_text
                state["waiting_for_cv"] = True

        if st.button("Analyze CV & Rank Jobs", type="primary",
                     disabled=st.session_state.cv_complete):
            with st.spinner("Analyzing your CV and ranking jobs..."):
                if (cv_method == "Upload File" and not state.get(
                        "cv_file_path")) or \
                        (cv_method == "Enter Text" and not state.get(
                            "user_input")):
                    st.error("Please provide your CV first")
                else:
                    # Create a unique session ID
                    session_id = f"streamlit-session-{int(time.time())}"
                    config = {"configurable": {"thread_id": session_id}}

                    try:
                        # Run the async code
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        state = loop.run_until_complete(
                            graph.ainvoke(state, config=config))

                        if state.get("ranked_jobs"):
                            # Debug information
                            st.success(
                                f"Successfully analyzed CV and ranked {len(state.get('ranked_jobs'))} jobs")

                            # Add debug output to check the first job's relevance data
                            if len(state.get("ranked_jobs")) > 0:
                                first_job = state.get("ranked_jobs")[0]
                                st.write("Debug - First job relevance info:")
                                st.write(
                                    f"Score: {first_job.get('relevance_score', 'Not found')}")
                                st.write(
                                    f"Explanation exists: {'Yes' if 'relevance_explanation' in first_job else 'No'}")
                                if 'relevance_explanation' in first_job:
                                    st.write(
                                        f"Explanation preview: {first_job['relevance_explanation'][:100]}...")

                                # Deep copy the ranked jobs to ensure we're not losing data
                                import copy

                                state["ranked_jobs"] = copy.deepcopy(
                                    state.get("ranked_jobs", []))

                            # Save the state with complete information
                            st.session_state.state = state
                            st.session_state.cv_complete = True
                            st.info(
                                "✓ CV analysis complete! Please proceed to the Results tab.")
                        else:
                            st.error(
                                "No jobs could be ranked. Please try again with a different CV or job search.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.error("Please try again with a different CV")

with tab4:
    st.header("Step 4: Results & CV Optimization")

    # Check if previous steps are complete
    if not st.session_state.cv_complete:
        st.warning("Please complete the CV analysis step first")
    else:
        state = st.session_state.state

        # Display ranked jobs
        ranked_jobs = state.get("ranked_jobs", [])
        if not ranked_jobs:
            st.error("No ranked jobs available")
        else:
            st.subheader("Top Matching Jobs")

            # Create columns for job selection checkboxes
            job_selection = []

            # Display top matching jobs without debug info
            for i, job in enumerate(ranked_jobs[:10], 1):
                # Make sure relevance score exists and is properly formatted
                relevance_score = job.get('relevance_score', 0)
                # Ensure it's a number for display purposes
                if not isinstance(relevance_score, (int, float)):
                    try:
                        relevance_score = int(relevance_score)
                    except (ValueError, TypeError):
                        relevance_score = 0

                expander = st.expander(
                    f"{i}. {job.get('job_title', 'Unknown Position')} at {job.get('company_name', 'Unknown Company')} - Match: {relevance_score}/100"
                )
                with expander:
                    cols = st.columns([1, 4])
                    with cols[0]:
                        selected = st.checkbox("Select", key=f"job_{i}")
                        if selected:
                            job_selection.append(i)

                    with cols[1]:
                        st.markdown(
                            f"**Location:** {job.get('location', 'Unknown')}")
                        st.markdown(
                            f"**Experience:** {job.get('experience_text', str(job.get('experience_years', 0)) + ' years')}")
                        st.markdown(
                            f"**Publication Date:** {job.get('publication_date', 'Unknown')}")
                        st.markdown(
                            f"**Job Type:** {job.get('job_type', 'Unknown')}")

                        st.markdown("**Relevance:**")
                        # Ensure the progress bar uses a valid value between 0 and 1
                        progress_value = max(0,
                                             min(relevance_score, 100)) / 100
                        st.progress(progress_value)

                        # Show all fields that might contain the explanation
                        explanation = job.get('relevance_explanation', '')
                        if not explanation:
                            explanation = job.get('explanation', '')

                        if explanation:
                            st.markdown(f"*{explanation}*")
                        else:
                            st.markdown("*No explanation available*")

                        # Show job description
                        st.markdown("**Description Preview:**")
                        description = job.get('description',
                                              'No description available')
                        st.markdown(description[:500] + (
                            '...' if len(description) > 500 else ''))

                        st.markdown("**Description Preview:**")
                        description = job.get('description',
                                              'No description available')
                        st.markdown(description[:500] + (
                            '...' if len(description) > 500 else ''))

                        if job.get('url'):
                            st.markdown(
                                f"[View Job Listing]({job.get('url')})")

            if job_selection:
                if st.button("Optimize CV for Selected Jobs", type="primary",
                             disabled=st.session_state.job_selection_complete):
                    with st.spinner(
                            "Optimizing your CV for the selected jobs..."):
                        # Update the state with the job selection
                        selection_str = ",".join(
                            [str(i) for i in job_selection])
                        state["job_selection_input"] = selection_str
                        state["waiting_for_job_selection"] = True

                        # Create a unique session ID
                        session_id = f"streamlit-session-{int(time.time())}"
                        config = {"configurable": {"thread_id": session_id}}

                        try:
                            # Run the async code
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            state = loop.run_until_complete(
                                graph.ainvoke(state, config=config))

                            if state.get("optimized_cvs"):
                                st.success(
                                    f"Successfully optimized CV for {len(state.get('optimized_cvs'))} jobs")
                                # Save the final results
                                st.session_state.results = state.get(
                                    "optimized_cvs")
                                st.session_state.job_selection_complete = True
                            else:
                                st.error(
                                    "Could not optimize CV. Please try again with different selections.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.info("Please select at least one job for CV optimization")

        # Display optimized CVs if available
        if st.session_state.job_selection_complete and st.session_state.results:
            st.subheader("CV Optimization Recommendations")

            for job_key, optimization in st.session_state.results.items():
                with st.expander(f"Optimized CV for: {job_key}"):
                    st.markdown(optimization)

                    # Add download button for this optimization
                    filename = f"optimized_cv_{job_key.replace(' ', '_').replace('/', '_')}.txt"
                    st.download_button(
                        label="Download as Text File",
                        data=optimization,
                        file_name=filename,
                        mime="text/plain"
                    )

            st.balloons()
            st.success(
                "🎉 Process complete! Your CV has been optimized for the selected jobs.")

            # Reset button
            if st.button("Start a New Search", type="secondary"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.experimental_rerun()