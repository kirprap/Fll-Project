import streamlit as st
import base64
import io
from PIL import Image
import pandas as pd
from datetime import datetime
from artifact_database import get_artifact_database
from ai_analyzer import analyze_artifact_image

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
                            # Add to history
                            add_to_history(image, result)
                            
                            # Display results
                            st.success("✅ Artifact identified successfully!")
                            
                            # Create columns for result display
                            result_col1, result_col2, result_col3 = st.columns(3)
                            
                            with result_col1:
                                st.metric("Artifact Name", result.get('name', 'Unknown'))
                            
                            with result_col2:
                                st.metric("Estimated Value", f"${result.get('value', 'Unknown')}")
                            
                            with result_col3:
                                st.metric("Age/Period", result.get('age', 'Unknown'))
                            
                            # Additional details
                            if result.get('description'):
                                st.subheader("Description")
                                st.write(result['description'])
                            
                            if result.get('cultural_context'):
                                st.subheader("Cultural Context")
                                st.write(result['cultural_context'])
                            
                            if result.get('confidence'):
                                st.subheader("Analysis Confidence")
                                confidence = float(result['confidence'])
                                st.progress(confidence)
                                st.write(f"Confidence: {confidence*100:.1f}%")
                        
                    except Exception as e:
                        st.error(f"❌ Error analyzing artifact: {str(e)}")
                        st.info("Please ensure you have a valid OpenAI API key set in your environment variables.")
    
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
                    if result.get('description'):
                        st.write(f"**Description:** {result['description']}")
        
        # Clear history button
        if st.button("🗑️ Clear Search History"):
            st.session_state.search_history = []
            st.rerun()
    
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
