import streamlit as st
import base64
import io
from PIL import Image
from datetime import datetime
from artifact_database import get_artifact_database
from ai_analyzer import analyze_artifact_image
from database import (save_artifact, get_all_artifacts, search_artifacts, 
    get_artifact_by_id, get_artifact_count, 
    update_artifact_verification, update_artifact_profile,
    get_artifacts_by_verification_status)
from ai_analyzer import compare_with_reference

# Configure page
st.set_page_config(
    page_title="Archaeological Artifact Identifier",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

def convert_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def convert_image_to_bytes(image):
    """Convert PIL image to bytes"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()

def add_to_history(image, result):
    """Add search result to session history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.search_history.append({
        'timestamp': timestamp,
        'image': image,
        'result': result
    })

def main():
    st.title("🏺 Archaeological Artifact Identifier")
    st.markdown("Upload photos of archaeological artifacts to identify them using AI vision technology.")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Identify New Artifact", "📦 Batch Upload", "📚 Archive", "✅ Expert Verification", "📊 Statistics"])
    
    with tab1:
        # Create two columns for main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Upload Artifact Photo")
            
            # File uploader with drag and drop
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear photo of your archaeological artifact. Supported formats: PNG, JPG, JPEG"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Artifact", use_column_width=True)
                
                # Analysis button
                if st.button("🔍 Identify Artifact", type="primary"):
                    with st.spinner("Analyzing artifact with AI vision..."):
                        try:
                            # Convert image to base64
                            base64_image = convert_image_to_base64(image)
                            
                            # Analyze with AI
                            result = analyze_artifact_image(base64_image)
                            
                            if result:
                                # Save to database
                                try:
                                    image_bytes = convert_image_to_bytes(image)
                                    saved_artifact = save_artifact(result, image_bytes)
                                    st.success(f"✅ Artifact identified and saved to archive! (ID: {saved_artifact.id})")
                                except Exception as db_error:
                                    st.warning(f"⚠️ Artifact identified but could not save to database: {str(db_error)}")
                                
                                # Add to session history
                                add_to_history(image, result)
                                
                                # Display results
                                
                                # Create columns for result display
                                result_col1, result_col2, result_col3 = st.columns(3)
                            

                                with result_col1:
                                    st.metric("Artifact Name", result.get('name', 'Unknown'))
                                
                                with result_col2:
                                    st.metric("Estimated Value", f"${result.get('value', 'Unknown')}")
                                
                                with result_col3:
                                    st.metric("Age/Period", result.get('age', 'Unknown'))
                                
                                # Additional details in expandable sections
                                with st.expander("📋 Detailed Information", expanded=True):
                                    if result.get('description'):
                                        st.subheader("Description")
                                        st.write(result['description'])
                                    
                                    if result.get('material'):
                                        st.subheader("Material")
                                        st.write(result['material'])
                                    
                                    if result.get('function'):
                                        st.subheader("Function/Purpose")
                                        st.write(result['function'])
                                    
                                    if result.get('rarity'):
                                        st.subheader("Rarity")
                                        st.write(result['rarity'])
                                
                                if result.get('cultural_context'):
                                    with st.expander("🌍 Cultural Context", expanded=False):
                                        st.write(result['cultural_context'])
                                
                                if result.get('confidence'):
                                    with st.expander("📊 Analysis Confidence", expanded=False):
                                        confidence = float(result['confidence'])
                                        st.progress(confidence)
                                        st.write(f"Confidence: {confidence*100:.1f}%")
                                
                                # Visual Similarity Search
                                st.write("---")
                                st.subheader("🔎 Similar Artifacts in Archive")
                                
                                try:
                                    # Get all artifacts for comparison (with images for AI)
                                    all_artifacts = get_all_artifacts(limit=20, include_images=True)
                                    
                                    if len(all_artifacts) > 1:  # More than just the current one
                                        with st.spinner("Searching for similar artifacts..."):
                                            # Use AI to compare with reference artifacts
                                            comparison_result = compare_with_reference(base64_image, all_artifacts)
                                            
                                            if comparison_result:
                                                closest_match = comparison_result.get('closest_match')
                                                similarity_score = comparison_result.get('similarity_score', 0)
                                                
                                                if closest_match and similarity_score > 0.3:
                                                    st.info(f"**Closest Match:** {closest_match} (Similarity: {similarity_score*100:.0f}%)")
                                                    
                                                    if comparison_result.get('differences'):
                                                        st.write(f"**Key Differences:** {comparison_result['differences']}")
                                                    
                                                    # Show alternative matches if available
                                                    alt_matches = comparison_result.get('alternative_matches', [])
                                                    if alt_matches:
                                                        st.write("**Other Possible Matches:**")
                                                        for match in alt_matches[:3]:
                                                            if isinstance(match, dict):
                                                                match_name = match.get('name', 'Unknown')
                                                                match_score = match.get('similarity', 0)
                                                                st.write(f"• {match_name} ({match_score*100:.0f}% similar)")
                                                else:
                                                    st.write("No significantly similar artifacts found in archive.")
                                            else:
                                                st.write("Unable to compare with archived artifacts.")
                                    else:
                                        st.write("No other artifacts in archive to compare with.")
                                
                                except Exception as sim_error:
                                    st.warning(f"Similarity search unavailable: {str(sim_error)}")
                            
                        except Exception as e:
                            st.error(f"❌ Error analyzing artifact: {str(e)}")
                            st.info("💡 AI analysis uses Hugging Face Inference. To use an authenticated client add a Hugging Face token named `HUGGINGFACE_TOKEN` in your environment or Secrets (Tools → Secrets → Add HUGGINGFACE_TOKEN). You can create a token at https://huggingface.co/settings/tokens. If no token is provided, the app will attempt unauthenticated public access which may be rate-limited.")
        
        with col2:
            st.header("Sample Artifacts")
            st.markdown("Reference database of common archaeological finds:")
            
            # Display sample artifact database
            artifact_db = get_artifact_database()
            
            for category, artifacts in artifact_db.items():
                with st.expander(f"📂 {category}"):
                    for artifact in artifacts:
                        st.write(f"**{artifact['name']}**")
                        st.write(f"Value: ${artifact['value']}")
                        st.write(f"Age: {artifact['age']}")
                        st.write(f"Description: {artifact['description']}")
                        st.write("---")
        
        # Search History Section
        if st.session_state.search_history:
            st.header("🕐 Search History")
            
            # Display search history
            for i, entry in enumerate(reversed(st.session_state.search_history)):
                with st.expander(f"Search {len(st.session_state.search_history) - i} - {entry['timestamp']}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(entry['image'], caption="Searched Image", width=200)
                    
                    with col2:
                        result = entry['result']
                        st.write(f"**Name:** {result.get('name', 'Unknown')}")
                        st.write(f"**Value:** ${result.get('value', 'Unknown')}")
                        st.write(f"**Age:** {result.get('age', 'Unknown')}")
                        if result.get('material'):
                            st.write(f"**Material:** {result['material']}")
                        if result.get('function'):
                            st.write(f"**Function:** {result['function']}")
                        if result.get('rarity'):
                            st.write(f"**Rarity:** {result['rarity']}")
                        if result.get('description'):
                            st.write(f"**Description:** {result['description']}")
            
                # Clear history button
                if st.button("🗑️ Clear Search History"):
                    st.session_state.search_history = []
                    st.rerun()
    
    with tab2:
        # Batch Upload Section
        st.header("📦 Batch Upload")
        st.markdown("Upload multiple artifact photos at once for efficient batch processing.")
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Select multiple artifact photos to process them all at once"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
            
            # Preview selected images
            if st.checkbox("Show preview of selected images"):
                preview_cols = st.columns(min(4, len(uploaded_files)))
                for idx, file in enumerate(uploaded_files[:4]):
                    with preview_cols[idx]:
                        img = Image.open(file)
                        st.image(img, caption=file.name, use_column_width=True)
                if len(uploaded_files) > 4:
                    st.caption(f"... and {len(uploaded_files) - 4} more")
            
            # Batch process button
            if st.button("🚀 Process All Artifacts", type="primary"):
                st.write("---")
                st.subheader("Processing Results")
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                successful = 0
                failed = 0
                results_list = []
                
                # Process each file
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Update progress
                        progress = (idx + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                        
                        # Process image
                        image = Image.open(uploaded_file)
                        base64_image = convert_image_to_base64(image)
                        
                        # Analyze with AI
                        result = analyze_artifact_image(base64_image)
                        
                        if result:
                            # Save to database
                            image_bytes = convert_image_to_bytes(image)
                            saved_artifact = save_artifact(result, image_bytes)
                            
                            results_list.append({
                                'filename': uploaded_file.name,
                                'status': 'success',
                                'artifact_id': saved_artifact.id,
                                'name': result.get('name', 'Unknown'),
                                'value': result.get('value', 'Unknown'),
                                'age': result.get('age', 'Unknown')
                            })
                            successful += 1
                        else:
                            results_list.append({
                                'filename': uploaded_file.name,
                                'status': 'failed',
                                'error': 'No result from AI'
                            })
                            failed += 1
                    
                    except Exception as e:
                        results_list.append({
                            'filename': uploaded_file.name,
                            'status': 'failed',
                            'error': str(e)
                        })
                        failed += 1
                
                # Complete progress
                progress_bar.progress(1.0)
                status_text.text(f"Processing complete!")
                
                # Show summary
                st.success(f"✅ Batch processing complete! Successfully processed {successful}/{len(uploaded_files)} artifacts")
                
                if failed > 0:
                    st.warning(f"⚠️ {failed} artifact(s) failed to process")
                
                st.write("---")
                st.subheader("Detailed Results")
                
                # Display results in expandable sections
                for result in results_list:
                    if result['status'] == 'success':
                        with st.expander(f"✅ {result['filename']} - {result['name']}"):
                            st.write(f"**Artifact ID:** {result['artifact_id']}")
                            st.write(f"**Name:** {result['name']}")
                            st.write(f"**Value:** ${result['value']}")
                            st.write(f"**Age:** {result['age']}")
                            st.success("Saved to archive")
                    else:
                        with st.expander(f"❌ {result['filename']} - Failed"):
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                
                st.info("💡 All successfully processed artifacts have been saved to the Archive. Visit the Archive tab to view them.")
    
    with tab3:
        # Archive Section
        st.header("📚 Artifact Archive")
        st.markdown("Browse and search all identified artifacts stored in the database.")
        
        try:
            # Search functionality
            search_col1, search_col2 = st.columns([3, 1])
            with search_col1:
                search_query = st.text_input("🔎 Search artifacts", placeholder="Search by name, material, description...")
            with search_col2:
                st.write("")
                st.write("")
                if st.button("Clear Search"):
                    search_query = ""
            
            # Get artifacts
            if search_query:
                artifacts = search_artifacts(search_query)
                st.info(f"Found {len(artifacts)} matching artifact(s)")
            else:
                artifacts = get_all_artifacts(limit=50)
                total_count = get_artifact_count()
                st.info(f"Showing {len(artifacts)} of {total_count} total artifacts")
            
            # Display artifacts
            if artifacts:
                for artifact in artifacts:
                    with st.expander(f"🏺 {artifact['name']} - ID: {artifact['id']}", expanded=False):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Display image from database
                            artifact_obj = get_artifact_by_id(artifact['id'])
                            if artifact_obj and artifact_obj.image_data:
                                st.image(artifact_obj.image_data, caption=artifact['name'], width=250)
                        
                        with col2:
                            # Display artifact details
                            st.subheader(artifact['name'])
                            
                            detail_col1, detail_col2, detail_col3 = st.columns(3)
                            with detail_col1:
                                st.metric("Value", f"${artifact['value']}")
                            with detail_col2:
                                st.metric("Age", artifact['age'])
                            with detail_col3:
                                st.metric("Confidence", f"{float(artifact['confidence'])*100:.0f}%")
                            
                            st.write(f"**Material:** {artifact['material']}")
                            st.write(f"**Function:** {artifact['function']}")
                            st.write(f"**Rarity:** {artifact['rarity']}")
                            st.write(f"**Verification Status:** {artifact['verification_status'].title()}")
                            

                            if artifact['description']:
                                st.write(f"**Description:** {artifact['description']}")
                            
                            if artifact['cultural_context']:
                                with st.expander("Cultural Context"):
                                    st.write(artifact['cultural_context'])
                            
                            uploaded_time = artifact.get('uploaded_at', 'Unknown')
                            st.caption(f"Uploaded: {uploaded_time}")
            else:
                st.info("No artifacts found. Start by identifying your first artifact in the 'Identify New Artifact' tab!")
        
        except ValueError as ve:
            st.error("⚠️ Database not configured")
            st.info("The archive feature requires a PostgreSQL database. The database should be automatically configured in your Replit environment.")
        except Exception as e:
            st.error(f"❌ Error accessing archive: {str(e)}")
            st.info("Please check your database configuration or try again later.")
    
    with tab4:
        # Expert Verification and Artifact Profiles Section
        st.header("✅ Expert Verification & Artifact Profiles")
        st.markdown("Review, verify, and add detailed information to identified artifacts.")
        
        try:
            # Sub-tabs for different functions
            verify_tab, profile_tab = st.tabs(["🔍 Verify Artifacts", "📝 Edit Artifact Profiles"])
            
            with verify_tab:
                st.subheader("Artifact Verification Queue")
                
                # Filter by verification status
                status_filter = st.selectbox(
                    "Filter by status",
                    ["all", "pending", "verified", "rejected"]
                )
                
                try:
                    if status_filter == "all":
                        artifacts_to_review = get_all_artifacts(limit=50)
                    else:
                        artifacts_to_review = get_artifacts_by_verification_status(status_filter, limit=50)
                except ValueError:
                    st.error("⚠️ Database not configured")
                    st.info("Expert verification requires a PostgreSQL database.")
                    artifacts_to_review = []
                
                if artifacts_to_review:
                    st.info(f"Found {len(artifacts_to_review)} artifact(s) to review")
                    
                    for artifact in artifacts_to_review:
                        with st.expander(f"ID: {artifact['id']} - {artifact['name']} [{artifact['verification_status'].upper()}]"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Display image
                                artifact_obj = get_artifact_by_id(artifact['id'])
                                if artifact_obj and artifact_obj.image_data:
                                    st.image(artifact_obj.image_data, caption=artifact['name'])
                            
                            with col2:
                                # Artifact details
                                st.write(f"**Name:** {artifact['name']}")
                                st.write(f"**Value:** ${artifact['value']}")
                                st.write(f"**Age:** {artifact['age']}")
                                st.write(f"**Material:** {artifact['material']}")
                                st.write(f"**Function:** {artifact['function']}")
                                st.write(f"**AI Confidence:** {float(artifact['confidence'])*100:.0f}%")
                                
                                st.write("---")
                                st.subheader("Verification Actions")
                                
                                # Verification form
                                with st.form(f"verify_form_{artifact['id']}"):
                                    verification_status = st.selectbox(
                                        "New Status",
                                        ["pending", "verified", "rejected"],
                                        index=["pending", "verified", "rejected"].index(artifact['verification_status'])
                                    )
                                    
                                    expert_name = st.text_input("Expert Name", value=artifact.get('verified_by', ''))
                                    comments = st.text_area("Verification Comments", value=artifact.get('verification_comments', ''))
                                    
                                    if st.form_submit_button("Update Verification"):
                                        try:
                                            update_artifact_verification(
                                                artifact['id'],
                                                verification_status,
                                                expert_name if expert_name else None,
                                                comments if comments else None
                                            )
                                            st.success(f"✅ Verification updated for artifact ID {artifact['id']}")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed to update: {str(e)}")
                else:
                    st.info("No artifacts found with the selected filter.")
            
            with profile_tab:
                st.subheader("Edit Artifact Profiles")
                
                # Select artifact to edit
                try:
                    all_artifacts = get_all_artifacts(limit=100)
                except ValueError:
                    st.error("⚠️ Database not configured")
                    st.info("Profile editing requires a PostgreSQL database.")
                    all_artifacts = []
                
                if all_artifacts:
                    artifact_options = {f"ID {a['id']}: {a['name']}": a['id'] for a in all_artifacts}
                    selected_artifact_label = st.selectbox("Select Artifact", list(artifact_options.keys()))
                    selected_artifact_id = artifact_options[selected_artifact_label]
                    
                    # Get full artifact details
                    artifact_obj = get_artifact_by_id(selected_artifact_id)
                    if artifact_obj:
                        artifact_dict = artifact_obj.to_dict()
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            if artifact_obj.image_data:
                                st.image(artifact_obj.image_data, caption=artifact_dict['name'])
                        
                        with col2:
                            st.write(f"**Basic Information:**")
                            st.write(f"Name: {artifact_dict['name']}")
                            st.write(f"Age: {artifact_dict['age']}")
                            st.write(f"Value: ${artifact_dict['value']}")
                        
                        st.write("---")
                        st.subheader("Detailed Profile Information")
                        
                        # Profile editing form
                        with st.form("profile_edit_form"):
                            st.write("**Add additional scholarly information:**")
                            
                            provenance = st.text_area(
                                "Provenance Information",
                                value=artifact_dict.get('provenance', ''),
                                help="Document the artifact's origin, ownership history, and discovery details"
                            )
                            
                            historical_context = st.text_area(
                                "Additional Historical Context",
                                value=artifact_dict.get('historical_context', ''),
                                help="Provide deeper historical context beyond the AI analysis"
                            )
                            
                            references = st.text_area(
                                "References & Citations",
                                value=artifact_dict.get('references', ''),
                                help="Add scholarly references, museum catalog numbers, or related publications"
                            )
                            
                            if st.form_submit_button("Save Profile Updates"):
                                try:
                                    update_artifact_profile(
                                        selected_artifact_id,
                                        provenance if provenance else None,
                                        historical_context if historical_context else None,
                                        references if references else None
                                    )
                                    st.success(f"✅ Profile updated for artifact ID {selected_artifact_id}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to update profile: {str(e)}")
                else:
                    st.info("No artifacts available. Identify some artifacts first!")
        
        except ValueError as ve:
            st.error("⚠️ Database not configured")
            st.info("Expert verification and profile management require a PostgreSQL database. The database should be automatically configured in your Replit environment.")
        except Exception as e:
            st.error(f"❌ Error accessing verification system: {str(e)}")
            st.info("Please check your database configuration or try again later.")
    
    with tab5:
        # Statistics Section
        st.header("📊 Archive Statistics")
        
        try:
            try:
                total_artifacts = get_artifact_count()
            except ValueError:
                st.error("⚠️ Database not configured")
                st.info("Statistics require a PostgreSQL database. The database should be automatically configured in your Replit environment.")
                total_artifacts = 0
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric("Total Artifacts", total_artifacts)
            
            with stat_col2:
                # Count by verification status
                pending = len(get_all_artifacts(limit=1000))  # Simplified for now
                st.metric("Total in Database", pending)
            
            with stat_col3:
                st.metric("Archive Size", f"{total_artifacts} items")
            
            st.markdown("---")
            st.subheader("Recent Identifications")
            
            recent = get_all_artifacts(limit=10)
            if recent:
                for artifact in recent:
                    st.write(f"• **{artifact['name']}** - {artifact['age']} (ID: {artifact['id']})")
                else:
                    st.info("No artifacts yet. Start identifying!")
        
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")
            if "DATABASE_URL" in str(e):
                st.info("Please ensure your database is properly configured.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("📖 About")
        st.write("""
        This AI-powered tool helps archaeologists identify artifacts from photographs using advanced computer vision technology.
        
        **Features:**
        - 📸 Photo upload with preview
        - 🤖 AI-powered artifact identification
        - 💰 Value estimation
        - 📅 Age/period determination
        - 📋 Search history tracking
        - 📱 Mobile-friendly interface
        """)
        
        st.header("🔧 Usage Tips")
        st.write("""
        For best results:
        - Use clear, well-lit photos
        - Show the artifact from multiple angles if possible
        - Include scale references when available
        - Ensure the artifact fills most of the frame
        """)
        
        st.header("⚠️ Disclaimer")
        st.write("""
        AI identifications are estimates and should be verified by professional archaeologists. Values are approximate and may vary based on condition, provenance, and market factors.
        """)

if __name__ == "__main__":
    main()