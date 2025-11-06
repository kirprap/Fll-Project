from typing import List, Dict, Any, Optional
import base64
from PIL import Image, ImageOps
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ai_analyzer import AIAnalyzer


# ============================================================================
# Streamlit UI
# ============================================================================

import streamlit as st
import io
from config import TIER_OPTIONS, TIER_KEY_MAP, TIER_INFO, EXPECTED_TIME

# Import database functions
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from database import save_artifact, get_all_artifacts, search_artifacts, get_artifact_by_id, init_db, update_artifact_tags
    # Initialize database tables on startup
    init_db()
except ModuleNotFoundError as e:
    if getattr(e, "name", None) == "database":
        st.error("Database module not found. Please ensure database.py is available.")
        st.stop()
    else:
        st.error(f"Missing dependency: {getattr(e, 'name', 'unknown')}. Please install requirements.")
        st.stop()
except Exception as e:
    st.error(f"Error importing database: {str(e)}")
    st.stop()


@st.cache_resource
def get_analyzer():
    """Cache the AI analyzer to avoid reloading the model."""
    return AIAnalyzer()


@st.cache_resource
def get_fast_analyzer(tier: str):
    """Cache the fast analyzer for the selected tier."""
    from fast_analyzer import FastAnalyzer
    return FastAnalyzer(tier=tier)


def _safe_rerun():
    """Rerun the app across Streamlit versions."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Archaeological Artifact Identifier",
        page_icon="🏺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏺 Artifact Gallery")

    # Query params helpers (support older/newer Streamlit)
    def get_qp():
        try:
            return dict(st.query_params)
        except Exception:
            return {k: v[0] if isinstance(v, list) and v else v for k, v in st.experimental_get_query_params().items()}

    def set_qp(**kwargs):
        try:
            st.query_params.clear()
            for k, v in kwargs.items():
                if v is not None:
                    st.query_params[k] = v
        except Exception:
            st.experimental_set_query_params(**{k: v for k, v in kwargs.items() if v is not None})

    # Session defaults
    if 'view_mode' not in st.session_state:
        st.session_state['view_mode'] = 'gallery'  # 'gallery' | 'add'

    # Sync selected artifact from query params
    qp = get_qp()
    selected_from_qp = qp.get('artifact')
    if selected_from_qp:
        st.session_state['selected_artifact'] = int(selected_from_qp)
        st.session_state['view_mode'] = 'gallery'

    # Top toolbar
    tb1, tb2, tb3, tb4 = st.columns([0.5, 6, 0.3, 0.3])
    with tb1:
        show_back = st.session_state.get('view_mode') == 'add' or ('selected_artifact' in st.session_state)
        if show_back and st.button('←', use_container_width=True):
            if 'selected_artifact' in st.session_state:
                del st.session_state['selected_artifact']
            st.session_state['view_mode'] = 'gallery'
            q = qp.get('q')
            set_qp(q=q)
            _safe_rerun()
    with tb2:
        search_val = qp.get('q', '')
        new_search = st.text_input('', value=search_val, placeholder='Search by name, description, material, or tags', label_visibility='collapsed')
    with tb3:
        if st.button('🔎', use_container_width=True):
            set_qp(q=new_search or None)
            _safe_rerun()
    with tb4:
        if st.button('➕', use_container_width=True):
            if 'selected_artifact' in st.session_state:
                del st.session_state['selected_artifact']
            st.session_state['view_mode'] = 'add'
    st.markdown(
        '<style>'
        'div[data-testid="stHorizontalBlock"] input {height:40px; margin-bottom:0;}'
        'div.stButton > button {height:40px; padding-top:0; padding-bottom:0; margin-bottom:0;}'
        '</style>',
        unsafe_allow_html=True
    )

    # Route views
    if st.session_state.get('view_mode') == 'add':
        identify_artifact_page()
    else:
        archive_page()


def identify_artifact_page():
    """Single artifact identification page."""
    st.header("Identify Single Artifact")
    
    # Choose source: Upload or Camera
    source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    uploaded_file = None
    camera_photo = None
    if source == "Upload":
        uploaded_file = st.file_uploader(
            "Upload an artifact image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, JPEG, PNG, WEBP"
        )
    else:
        camera_photo = st.camera_input("Take a picture (Identify page)", key="identify_camera")

    # Resolve to a PIL image if provided
    image_input = None
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)
    elif camera_photo is not None:
        image_input = Image.open(camera_photo)

    if image_input is not None:
        # Clear previous results if a different file is uploaded
        file_key = getattr(uploaded_file, 'name', None) or 'camera_capture'
        if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != file_key:
            st.session_state['last_uploaded_file'] = file_key
            if 'last_analysis' in st.session_state:
                del st.session_state['last_analysis']
            if 'last_image' in st.session_state:
                del st.session_state['last_image']
        
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            image = image_input
            st.image(image, use_container_width=True)

            # Optional cropping UI
            st.markdown("**Crop Options**")
            enable_crop = st.checkbox("Crop image", value=False)
            if enable_crop:
                w, h = image.size
                colc1, colc2 = st.columns(2)
                with colc1:
                    left_pct = st.slider("Left %", 0, 99, 0)
                    top_pct = st.slider("Top %", 0, 99, 0)
                with colc2:
                    right_pct = st.slider("Right %", 1, 100, 100)
                    bottom_pct = st.slider("Bottom %", 1, 100, 100)

                # Ensure valid box
                left = int((left_pct/100.0) * w)
                top = int((top_pct/100.0) * h)
                right = int((right_pct/100.0) * w)
                bottom = int((bottom_pct/100.0) * h)
                if right <= left:
                    right = left + 1 if left < w else w
                if bottom <= top:
                    bottom = top + 1 if top < h else h

                try:
                    cropped = image.crop((left, top, right, bottom))
                except Exception:
                    cropped = image
                st.image(cropped, caption="Cropped Preview", use_container_width=True)
                image_to_use = cropped
            else:
                image_to_use = image

        with col2:
            st.subheader("Analysis Results")

            # Speed tier selection
            st.markdown("**⚡ Speed vs Quality**")
            speed_tier = st.radio(
                "Choose analysis speed:",
                TIER_OPTIONS,
                index=1,  # Default to FAST
                help="Faster = less detailed, Slower = more detailed",
                horizontal=True
            )

            # Extract tier name
            selected_tier = TIER_KEY_MAP[speed_tier]

            # Show what this tier uses
            st.caption(TIER_INFO[selected_tier])

            if st.button("Analyze Artifact", type="primary"):
                expected_time = EXPECTED_TIME[selected_tier]

                with st.spinner(f"Analyzing artifact... Expected time: {expected_time}"):
                    try:
                        # Use fast analyzer
                        analyzer = get_fast_analyzer(selected_tier)
                        result = analyzer.analyze_artifact(image_to_use)
                        
                        # Store results in session state for persistence
                        st.session_state['last_analysis'] = result
                        st.session_state['last_image'] = image_to_use

                    except RuntimeError as e:
                        error_msg = str(e)
                        if "Ollama generation failed" in error_msg or "timeout" in error_msg.lower():
                            st.error("⚠️ **Ollama Connection Error**")
                            st.markdown("""
                            **Possible causes:**
                            1. **Ollama is not running** - Start Ollama service
                            2. **Model not downloaded** - Run: `ollama pull qwen3-vl:32b`
                            3. **Timeout (model too large)** - The model is processing, please wait
                            4. **Wrong endpoint** - Check OLLAMA_ENDPOINT environment variable

                            **Quick fixes:**
                            - **Docker**: `docker-compose restart ollama`
                            - **Local**: `ollama serve` in a terminal
                            - **Check model**: `ollama list`
                            - **Pull model**: `ollama pull qwen3-vl:32b`

                            **Alternative**: Try using the **ViT** model instead (faster, no Ollama required)
                            """)
                            with st.expander("Technical Details"):
                                st.code(error_msg)
                        else:
                            st.error(f"Error during analysis: {error_msg}")
                            st.exception(e)
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
                        st.exception(e)
            
            # Display results if available (persists across reruns)
            if 'last_analysis' in st.session_state:
                result = st.session_state['last_analysis']
                image = st.session_state['last_image']
                
                st.success(f"✅ Analysis Complete in {result.get('analysis_time', 'N/A')}!")
                st.markdown(f"**Name:** {result.get('name', 'Unknown')}")
                st.markdown(f"**Description:** {result.get('description', 'N/A')}")
                st.markdown(f"**Confidence:** {result.get('confidence', 0):.2%}")
                st.markdown(f"**Method:** {result.get('method', 'N/A')}")
                st.markdown(f"**Quality Tier:** {result.get('tier', 'N/A')}")

                # Option to save to archive (now outside the analyze button block)
                tags_input = st.text_input("Tags (comma-separated)", placeholder="e.g. pottery, bronze, burial")
                if st.button("Save to Archive"):
                    try:
                        img_bytes = io.BytesIO()
                        # Use cropped/selected image
                        image_to_save = st.session_state.get('last_image', image_to_use)
                        image_to_save.save(img_bytes, format='PNG')
                        artifact_data = {
                            "name": result.get('name', 'Unknown'),
                            "description": result.get('description', ''),
                            "confidence": result.get('confidence', 0.0),
                            "value": "Requires expert appraisal",
                            "age": "Requires expert analysis",
                            "cultural_context": "Requires expert input",
                            "material": "Requires physical inspection",
                            "function": "Inferred from analysis",
                            "rarity": "Requires comparison",
                            "tags": [t.strip() for t in tags_input.split(',') if t.strip()] if tags_input else None,
                        }
                        artifact_id = save_artifact(artifact_data, img_bytes.getvalue())
                        st.success(f"✅ Artifact saved to archive with ID: {artifact_id}")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error saving artifact: {str(e)}")
                        st.exception(e)


def _make_square_thumbnail_b64(img_b64: str, size: int = 300) -> str:
    """Create a square thumbnail from a base64 image string and return base64 PNG."""
    try:
        img_data = base64.b64decode(img_b64)
        with Image.open(io.BytesIO(img_data)) as im:
            im = im.convert('RGBA')
            # Fit into square with padding
            thumb = ImageOps.contain(im, (size, size))
            canvas = Image.new('RGBA', (size, size), (255, 255, 255, 0))
            x = (size - thumb.width) // 2
            y = (size - thumb.height) // 2
            canvas.paste(thumb, (x, y))
            out = io.BytesIO()
            canvas.save(out, format='PNG')
            return base64.b64encode(out.getvalue()).decode('utf-8')
    except Exception:
        return img_b64


def archive_page():
    """Archive page to view saved artifacts."""
    st.header("Gallery")

    try:
        # Determine search from query params
        try:
            qp = dict(st.query_params)
        except Exception:
            qp = {k: v[0] if isinstance(v, list) and v else v for k, v in st.experimental_get_query_params().items()}
        search_q = (qp.get('q') or '').strip()
        artifact_qp = qp.get('artifact')
        if artifact_qp and 'selected_artifact' not in st.session_state:
            try:
                st.session_state['selected_artifact'] = int(artifact_qp)
            except Exception:
                pass

        # If an artifact is selected, show only its detail page
        if 'selected_artifact' in st.session_state:
            artifact_id = st.session_state['selected_artifact']
            artifact = get_artifact_by_id(artifact_id)
            if artifact:
                st.subheader(artifact.get('name', 'Artifact'))
                left, right = st.columns([1, 2])
                with left:
                    if artifact.get('image_data'):
                        img = Image.open(io.BytesIO(artifact['image_data']))
                        st.image(img, use_container_width=True)
                with right:
                    st.markdown(f"**Description:** {artifact.get('description') or 'N/A'}")
                    colA, colB, colC = st.columns(3)
                    with colA:
                        st.markdown(f"**Material:** {artifact.get('material') or 'N/A'}")
                        st.markdown(f"**Age:** {artifact.get('age') or 'N/A'}")
                    with colB:
                        st.markdown(f"**Cultural Context:** {artifact.get('cultural_context') or 'N/A'}")
                        st.markdown(f"**Function:** {artifact.get('function') or 'N/A'}")
                    with colC:
                        st.markdown(f"**Rarity:** {artifact.get('rarity') or 'N/A'}")
                        st.markdown(f"**Value:** {artifact.get('value') or 'N/A'}")
                    confidence = artifact.get('confidence')
                    st.markdown(f"**Confidence:** {confidence:.2%}" if confidence else "**Confidence:** N/A")
                    st.markdown(f"**Uploaded:** {artifact.get('uploaded_at', 'N/A')}")

                # Clickable tags that trigger search
                tags_str = artifact.get('tags') or ''
                tags = [t.strip() for t in tags_str.split(',') if t.strip()]
                if tags:
                    st.markdown("**Tags:**")
                    tcols = st.columns(min(4, len(tags)))
                    for idx, tag in enumerate(tags):
                        with tcols[idx % len(tcols)]:
                            if st.button(f"#{tag}", key=f"tag_{artifact_id}_{idx}"):
                                try:
                                    st.query_params['q'] = tag
                                except Exception:
                                    st.experimental_set_query_params(q=tag)
                                # Clear selection to show search results
                                if 'selected_artifact' in st.session_state:
                                    del st.session_state['selected_artifact']
                                _safe_rerun()
                
                # Tag editing
                edit_tags = st.text_input("Edit Tags (comma-separated)", value=artifact.get('tags') or '', key=f"tags_edit_{artifact_id}")
                if st.button("Update Tags", key=f"update_tags_{artifact_id}"):
                    try:
                        new_tags_list = [t.strip() for t in edit_tags.split(',') if t.strip()]
                        updated = update_artifact_tags(artifact_id, new_tags_list)
                        if updated:
                            st.session_state['selected_artifact'] = artifact_id
                            st.success("Tags updated.")
                        else:
                            st.warning("Artifact not found.")
                    except Exception as e:
                        st.error(f"Failed to update tags: {str(e)}")
            return

        # Otherwise, show gallery grid (optionally filtered)
        if search_q:
            results = search_artifacts(search_q, limit=50, tags=[search_q])
            # search_artifacts may not include images; fetch per id to ensure image
            artifacts = []
            for r in results:
                full = get_artifact_by_id(r.get('id'))
                if full:
                    # add base64 for grid
                    if not full.get('image_base64') and full.get('image_data'):
                        try:
                            b64 = base64.b64encode(full['image_data']).decode('utf-8')
                            full['image_base64'] = b64
                        except Exception:
                            pass
                    artifacts.append(full)
        else:
            artifacts = get_all_artifacts(limit=50, include_images=True)

        if not artifacts:
            st.info("No artifacts in archive yet. Use + to add your first artifact.")
            return

        cols_per_row = 3
        for i in range(0, len(artifacts), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(artifacts):
                    artifact = artifacts[i + j]
                    with col:
                        with st.container():
                            # Clickable square thumbnail
                            if artifact.get('image_base64'):
                                thumb_b64 = _make_square_thumbnail_b64(artifact['image_base64'], size=300)
                                st.markdown(
                                    f'<a href="?artifact={artifact.get("id")}" target="_self"><img src="data:image/png;base64,{thumb_b64}" style="width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;border:1px solid #ccc" /></a>',
                                    unsafe_allow_html=True
                                )
                            st.markdown(f"**{artifact.get('name', 'Unknown')}**")
                            st.caption(f"ID: {artifact.get('id')} | Uploaded: {artifact.get('uploaded_at', 'N/A')}")
                            tags_str = artifact.get('tags') or ''
                            tags = [t.strip() for t in tags_str.split(',') if t.strip()]
                            if tags:
                                st.markdown("**Tags:**")
                                tcols = st.columns(min(4, len(tags)))
                                for idx, tag in enumerate(tags):
                                    with tcols[idx % len(tcols)]:
                                        if st.button(f"#{tag}", key=f"grid_tag_{artifact.get('id')}_{idx}"):
                                            try:
                                                st.query_params['q'] = tag
                                            except Exception:
                                                st.experimental_set_query_params(q=tag)
                                            if 'selected_artifact' in st.session_state:
                                                del st.session_state['selected_artifact']
                                            _safe_rerun()

    except Exception as e:
        st.error(f"Error loading archive: {str(e)}")
        st.exception(e)


def search_page():
    """Search page for finding artifacts."""
    st.header("Search Artifacts")

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        search_query = st.text_input(
            "Search",
            placeholder="Enter keywords to search...",
            help="Search by name, description, material, or cultural context"
        )
    with col2:
        tags_filter_input = st.text_input(
            "Tags",
            placeholder="e.g. pottery, bronze"
        )
    with col3:
        do_search = st.button("Search")

    tags_filter_list = [t.strip() for t in tags_filter_input.split(',') if t.strip()] if tags_filter_input else None

    if do_search or search_query or tags_filter_list:
        try:
            results = search_artifacts(search_query or "", limit=20, tags=tags_filter_list)

            if results:
                st.success(f"Found {len(results)} matching artifacts")

                for artifact in results:
                    with st.expander(f"🏺 {artifact.get('name', 'Unknown')} (ID: {artifact.get('id')})"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown(f"**ID:** {artifact.get('id')}")
                            st.markdown(f"**Uploaded:** {artifact.get('uploaded_at', 'N/A')}")
                        with col2:
                            st.markdown(f"**Description:** {artifact.get('description', 'N/A')}")
                            st.markdown(f"**Material:** {artifact.get('material', 'N/A')}")
                            st.markdown(f"**Cultural Context:** {artifact.get('cultural_context', 'N/A')}")
                            st.markdown(f"**Tags:** {artifact.get('tags') or 'N/A'}")
            else:
                st.info("No artifacts found matching your search.")

        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
