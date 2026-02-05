import base64
import logging
import os
import tempfile
from io import BytesIO

import obj2html
import requests
import streamlit as st
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3D Model Viewer import (optional - gracefully handle if not available)
try:
    from model_3d_viewer import Model3DViewer, save_uploaded_obj_file

    HAS_3D_VIEWER = True
except ImportError:
    HAS_3D_VIEWER = False
    logger.warning("3D viewer module not available")

# API configuration
API_BASE_URL = "http://localhost:8000"


# API Client Functions
def api_request(method, endpoint, data=None, params=None):
    """Generic API request helper"""
    url = f"{API_BASE_URL}/{endpoint}"
    headers = {"Content-Type": "application/json"}

    # Debug logging
    if data:
        logger.info(f"API {method} {endpoint} with data: {data}")

    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, headers=headers)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()
        return response.json() if response.content else {}
    except requests.exceptions.HTTPError as e:
        # Try to get error details from response
        try:
            error_detail = e.response.json().get("detail", str(e))
        except:
            error_detail = str(e)
        logger.error(f"API request failed: {error_detail}")
        raise Exception(f"API Error: {error_detail}")
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise


def get_artifacts():
    """Get all artifacts"""
    return api_request("GET", "api/artifacts")


def search_artifacts(query):
    """Search artifacts by query"""
    return api_request("GET", "api/artifacts/search", params={"q": query})


def get_artifact(artifact_id):
    """Get a single artifact by ID"""
    return api_request("GET", f"api/artifacts/{artifact_id}")


def create_artifact(
    name,
    description,
    tags,
    tier,
    image_data,
    form_data=None,
    model_3d_data=None,
    model_3d_format=None,
    uploaded_by=None,
):
    """Create a new artifact with optional 3D model"""
    payload = {
        "name": name,
        "description": description,
        "tags": tags,
        "tier": tier,
        "image_data": image_data,
    }
    if form_data:
        payload["form_data"] = form_data
    if model_3d_data:
        payload["model_3d_data"] = model_3d_data
        payload["model_3d_format"] = model_3d_format or "obj"
    if uploaded_by:
        payload["uploaded_by"] = uploaded_by
    return api_request("POST", "api/artifacts", data=payload)


def analyze_image(image_data, tier="fast"):
    """Analyze an image using the AI analyzer"""
    payload = {
        "image_data": image_data,
        "tier": tier,
    }
    return api_request("POST", "api/analyze", data=payload)


def main(role):
    """Main Streamlit application."""
    st.title("🏺 Artifact Gallery")

    # Store role in session state for access in other components
    st.session_state["role"] = role

    # Query params helper functions
    def get_query_params():
        """Helper function to get query params"""
        try:
            return dict(st.query_params)
        except Exception:
            return {
                k: v[0] if isinstance(v, list) and v else v
                for k, v in st.experimental_get_query_params().items()
            }

    def set_qp(**kwargs):
        try:
            st.query_params.clear()
            for k, v in kwargs.items():
                if v is not None:
                    st.query_params[k] = v
        except Exception:
            st.experimental_set_query_params(
                **{k: v for k, v in kwargs.items() if v is not None}
            )

    # Session state defaults
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "gallery"

    # Sync selected artifact from query params
    qp = get_query_params()
    selected_from_qp = str(qp.get("artifact", ""))
    if selected_from_qp:
        try:
            st.session_state["selected_artifact"] = int(selected_from_qp)
            st.session_state["view_mode"] = "gallery"
        except (ValueError, TypeError):
            pass

    # Top toolbar
    tb1, tb2, tb3, tb4 = st.columns([0.5, 6, 0.3, 0.3])

    with tb1:
        show_back = st.session_state.get("view_mode") == "add" or (
            "selected_artifact" in st.session_state
        )
        if show_back and st.button("←", use_container_width=True):
            if "selected_artifact" in st.session_state:
                del st.session_state["selected_artifact"]
            st.session_state["view_mode"] = "gallery"
            q = qp.get("q")
            set_qp(q=q)
            st.rerun()

    with tb2:
        if role in ["admin", "field", "lab"]:
            search_val = qp.get("q", "")
            new_search = st.text_input(
                "Search",
                value=search_val,
                placeholder="Search by name, description, material, or tags",
                label_visibility="collapsed",
            )
        else:
            new_search = ""

    with tb3:
        if role in ["admin", "field", "lab"]:
            if st.button("🔎", use_container_width=True):
                set_qp(q=new_search or None)
                st.rerun()

    with tb4:
        if st.button("➕", use_container_width=True):
            if "selected_artifact" in st.session_state:
                del st.session_state["selected_artifact"]
            st.session_state["view_mode"] = "add"
            st.rerun()

    st.markdown(
        "<style>"
        'div[data-testid="stHorizontalBlock"] input {height:40px; margin-bottom:0;}'
        "div.stButton > button {height:40px; padding-top:0; padding-bottom:0; margin-bottom:0;}"
        "</style>",
        unsafe_allow_html=True,
    )

    # Route views
    if st.session_state.get("view_mode") == "add":
        identify_artifact_page()
    else:
        if role != "user":
            archive_page()
        else:
            st.info("Upload artifacts or browse the gallery from the menu.")


def identify_artifact_page():
    """Artifact identification page with form data capture - supports single or batch upload."""
    st.header("📤 Upload & Identify Artifact")

    # Choose upload mode: Single (with form) or Batch (multiple images)
    upload_mode = st.radio(
        "Upload mode",
        ["Single (with details)", "Batch (multiple images)"],
        horizontal=True,
        help="Single mode allows detailed form data per artifact. Batch mode analyzes multiple images quickly.",
    )

    if upload_mode == "Batch (multiple images)":
        # Batch upload mode
        _batch_upload_section()
        return

    # Single artifact mode
    # Choose source: Upload or Camera
    source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    uploaded_file = None
    camera_photo = None

    if source == "Upload":
        uploaded_file = st.file_uploader(
            "Upload an artifact image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, JPEG, PNG, WEBP",
        )
    else:
        camera_photo = st.camera_input(
            "Take a picture of the artifact", key="identify_camera"
        )

    # Resolve to a PIL image if provided
    image_input = None
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)
    elif camera_photo is not None:
        image_input = Image.open(camera_photo)

    if image_input is not None:
        # Clear previous results if a different file is uploaded
        file_key = getattr(uploaded_file, "name", None) or "camera_capture"
        if (
            "last_uploaded_file" not in st.session_state
            or st.session_state["last_uploaded_file"] != file_key
        ):
            st.session_state["last_uploaded_file"] = file_key
            if "last_analysis" in st.session_state:
                del st.session_state["last_analysis"]
            if "last_form_data" in st.session_state:
                del st.session_state["last_form_data"]

        # Create two columns: Image & Form on left, Analysis on right
        col1, col2 = st.columns([1, 1])

        # LEFT COLUMN: Image and Form Data
        with col1:
            st.subheader("📷 Artifact Image")
            st.image(image_input, use_container_width=True)

            # Cropping UI
            st.markdown("**Crop Options**")
            enable_crop = st.checkbox("Crop image", value=False)
            image_to_use = image_input

            if enable_crop:
                w, h = image_input.size
                colc1, colc2 = st.columns(2)
                with colc1:
                    left_pct = st.slider("Left %", 0, 99, 0)
                    top_pct = st.slider("Top %", 0, 99, 0)
                with colc2:
                    right_pct = st.slider("Right %", 1, 100, 100)
                    bottom_pct = st.slider("Bottom %", 1, 100, 100)

                # Ensure valid box
                left = int((left_pct / 100.0) * w)
                top = int((top_pct / 100.0) * h)
                right = int((right_pct / 100.0) * w)
                bottom = int((bottom_pct / 100.0) * h)
                if right <= left:
                    right = left + 1 if left < w else w
                if bottom <= top:
                    bottom = top + 1 if top < h else h

                try:
                    image_to_use = image_input.crop((left, top, right, bottom))
                    st.image(
                        image_to_use,
                        caption="Cropped Preview",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"Crop failed: {str(e)}")
                    image_to_use = image_input

            # Form for capturing artifact details
            st.markdown("**📋 Artifact Details**")
            with st.form("artifact_details_form"):
                st.subheader("Physical Measurements")
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    length = st.number_input(
                        "Length (cm)", min_value=0.0, step=0.1, value=0.0
                    )

                with col_b:
                    width = st.number_input(
                        "Width (cm)", min_value=0.0, step=0.1, value=0.0
                    )

                with col_c:
                    thickness = st.number_input(
                        "Thickness (cm)", min_value=0.0, step=0.1, value=0.0
                    )

                color = st.color_picker("Artifact Color", "#808080")
                location = st.text_input(
                    "Discovery Location", placeholder="e.g., Site A, Grid 5"
                )
                description = st.text_area(
                    "Physical Description",
                    placeholder="Describe the artifact's appearance, condition, material, etc.",
                )

                artifact_name = st.text_input(
                    "Artifact Name (Optional)",
                    placeholder="e.g., Pottery Fragment, Bronze Tool",
                )
                tags = st.text_input(
                    "Tags (comma-separated)",
                    placeholder="e.g., pottery, bronze, burial, broken",
                )

                submitted = st.form_submit_button(
                    "💾 Save Form Data", use_container_width=True
                )

            if submitted:
                # Validate form data
                if length == 0.0 and width == 0.0 and thickness == 0.0:
                    st.warning("⚠️ Please enter at least one measurement")
                elif not location.strip():
                    st.warning("⚠️ Please enter a discovery location")
                elif not description.strip():
                    st.warning("⚠️ Please provide a physical description")
                else:
                    # Store form data in session state
                    form_data = {
                        "length": float(length),
                        "width": float(width),
                        "thickness": float(thickness),
                        "color": color,
                        "location": location.strip(),
                        "description": description.strip(),
                        "artifact_name": artifact_name.strip()
                        if artifact_name
                        else None,
                        "tags": [t.strip() for t in tags.split(",") if t.strip()]
                        if tags
                        else [],
                    }
                    st.session_state["last_form_data"] = form_data
                    st.success("✅ Form data saved! You can now analyze the artifact.")

            # 3D Model Upload Section (outside form due to Streamlit limitations)
            st.markdown("---")
            st.markdown("**🎯 3D Model (Optional)**")
            uploaded_3d_model = st.file_uploader(
                "Upload 3D Model",
                type=["obj", "stl", "ply"],
                help="Optional: Upload a 3D model scan of the artifact (OBJ, STL, or PLY format)",
                key="model_3d_upload",
            )

            if uploaded_3d_model is not None:
                # Store 3D model in session state
                model_3d_bytes = uploaded_3d_model.getvalue()
                model_3d_b64 = base64.b64encode(model_3d_bytes).decode()
                model_format = uploaded_3d_model.name.split(".")[-1].lower()
                st.session_state["last_model_3d_data"] = model_3d_b64
                st.session_state["last_model_3d_format"] = model_format
                st.success(
                    f"✅ 3D model loaded: {uploaded_3d_model.name} ({model_format.upper()})"
                )

                # Preview 3D model if viewer is available
                if HAS_3D_VIEWER and model_format == "obj":
                    with st.expander("🔍 Preview 3D Model", expanded=False):
                        temp_file_path = save_uploaded_obj_file(uploaded_3d_model)
                        if temp_file_path:
                            try:
                                viewer = Model3DViewer()
                                viewer.render_3d_model(temp_file_path, height=400)
                            except Exception as e:
                                st.warning(f"Could not render 3D preview: {e}")
                            finally:
                                try:
                                    os.unlink(temp_file_path)
                                except:
                                    pass
            else:
                # Clear 3D model from session if removed
                if "last_model_3d_data" in st.session_state:
                    del st.session_state["last_model_3d_data"]
                if "last_model_3d_format" in st.session_state:
                    del st.session_state["last_model_3d_format"]

        # RIGHT COLUMN: Analysis Controls and Results
        with col2:
            st.subheader("🤖 AI Analysis")

            # Check if form data is saved
            form_saved = "last_form_data" in st.session_state

            if not form_saved:
                st.info("💡 Complete the form on the left before analyzing")

            # Speed tier selection
            tier = st.selectbox(
                "Analysis Quality",
                ["fast", "balanced", "thorough"],
                index=0,
                help="Fast: ~20-40s | Balanced: ~30-60s | Thorough: ~1-2 min",
            )

            # Analyze button
            if st.button(
                "🔍 Analyze Artifact", disabled=not form_saved, use_container_width=True
            ):
                with st.spinner("🔄 Analyzing artifact with AI..."):
                    try:
                        # Convert image to base64
                        buffered = BytesIO()
                        image_to_use.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        img_data_url = f"data:image/png;base64,{img_str}"

                        # Call analyze endpoint
                        analysis = analyze_image(img_data_url, tier=tier)
                        st.session_state["last_analysis"] = analysis
                        st.session_state["last_image"] = image_to_use
                        st.success("✅ Analysis complete!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        logger.exception("Analysis error")

            # Display analysis results if available
            if "last_analysis" in st.session_state:
                result = st.session_state["last_analysis"]
                st.markdown("---")
                st.markdown("**Analysis Results:**")

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric(
                        "Confidence", f"{float(result.get('confidence', 0)) * 100:.1f}%"
                    )
                    st.metric("Tier", result.get("tier", "N/A").title())

                with col_res2:
                    st.metric("Method", result.get("method", "N/A"))
                    st.metric("Time", result.get("analysis_time", "N/A"))

                st.markdown(f"**Name:** {result.get('name', 'Unknown')}")
                st.markdown(
                    f"**Description:** {result.get('description', 'No description')}"
                )

                # Save to archive section
                st.markdown("---")
                st.markdown("**📦 Save to Archive**")

                with st.form("save_to_archive_form"):
                    final_name = st.text_input(
                        "Artifact Name", value=result.get("name", "Unknown")
                    )
                    final_description = st.text_area(
                        "Description", value=result.get("description", ""), height=100
                    )
                    final_tags = st.text_input(
                        "Tags (comma-separated)", value=",".join(result.get("tags", []))
                    )

                    if st.form_submit_button(
                        "✅ Save to Archive", use_container_width=True
                    ):
                        try:
                            # Get the last form data
                            form_data = st.session_state.get("last_form_data", {})

                            # Prepare image data
                            buffered = BytesIO()
                            st.session_state["last_image"].save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            img_data_url = f"data:image/png;base64,{img_str}"

                            # Prepare tags
                            tag_list = [
                                t.strip() for t in final_tags.split(",") if t.strip()
                            ]

                            # Get 3D model data if available
                            model_3d_data = st.session_state.get("last_model_3d_data")
                            model_3d_format = st.session_state.get(
                                "last_model_3d_format"
                            )

                            # Create artifact with form data and 3D model attached
                            result_data = create_artifact(
                                name=final_name,
                                description=final_description,
                                tags=tag_list,
                                tier=tier,
                                image_data=img_data_url,
                                form_data=form_data,
                                model_3d_data=model_3d_data,
                                model_3d_format=model_3d_format,
                                uploaded_by=st.session_state.get("username"),
                            )

                            artifact_id = result_data.get("id")
                            has_3d = result_data.get("has_3d_model", False)
                            save_msg = f"✅ Artifact saved! ID: {artifact_id}"
                            if has_3d:
                                save_msg += " (with 3D model)"
                            st.success(save_msg)
                            st.balloons()

                            # Clear session state
                            if "last_analysis" in st.session_state:
                                del st.session_state["last_analysis"]
                            if "last_form_data" in st.session_state:
                                del st.session_state["last_form_data"]
                            if "last_image" in st.session_state:
                                del st.session_state["last_image"]
                            if "last_model_3d_data" in st.session_state:
                                del st.session_state["last_model_3d_data"]
                            if "last_model_3d_format" in st.session_state:
                                del st.session_state["last_model_3d_format"]

                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Failed to save artifact: {str(e)}")
                            logger.exception("Save artifact error")


def _batch_upload_section():
    """Batch upload section for multiple images."""
    st.subheader("📦 Batch Upload")
    st.info(
        "Upload multiple artifact images for quick analysis. Each image will be analyzed and can be saved individually."
    )

    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Upload artifact images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Select multiple images to analyze in batch",
        key="batch_upload",
    )

    if not uploaded_files:
        st.info("👆 Select one or more images to begin batch analysis")
        return

    st.success(f"✅ {len(uploaded_files)} image(s) selected")

    # Tier selection for batch
    tier = st.selectbox(
        "Analysis Quality",
        ["fast", "balanced", "thorough"],
        index=0,
        help="Fast: ~20-40s per image | Balanced: ~30-60s | Thorough: ~1-2 min",
    )

    # Optional 3D model for batch (applies to all artifacts in batch)
    st.markdown("---")
    st.markdown("**🎯 3D Model (Optional - applies to all artifacts)**")
    batch_3d_model = st.file_uploader(
        "Upload 3D Model for batch",
        type=["obj", "stl", "ply"],
        help="Optional: This 3D model will be attached to all artifacts in this batch",
        key="batch_model_3d_upload",
    )

    batch_model_3d_data = None
    batch_model_3d_format = None
    if batch_3d_model:
        batch_model_3d_data = base64.b64encode(batch_3d_model.getvalue()).decode()
        batch_model_3d_format = batch_3d_model.name.split(".")[-1].lower()
        st.success(f"✅ 3D model loaded: {batch_3d_model.name}")

    # Analyze button
    if st.button("🔍 Analyze All Images", use_container_width=True):
        # Initialize batch results in session state
        st.session_state["batch_results"] = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(
                f"Analyzing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}"
            )
            progress_bar.progress((idx) / len(uploaded_files))

            try:
                # Convert to base64
                image = Image.open(uploaded_file)
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_data_url = f"data:image/png;base64,{img_str}"

                # Analyze
                result = analyze_image(img_data_url, tier=tier)

                # Store result with image data
                result["_image_data"] = img_data_url
                result["_filename"] = uploaded_file.name
                result["_model_3d_data"] = batch_model_3d_data
                result["_model_3d_format"] = batch_model_3d_format
                st.session_state["batch_results"].append(result)

            except Exception as e:
                st.session_state["batch_results"].append(
                    {
                        "error": str(e),
                        "_filename": uploaded_file.name,
                        "name": f"Error: {uploaded_file.name}",
                    }
                )

        progress_bar.progress(1.0)
        status_text.text("✅ Batch analysis complete!")

    # Display batch results
    if "batch_results" in st.session_state and st.session_state["batch_results"]:
        st.markdown("---")
        st.subheader("📊 Batch Results")

        results = st.session_state["batch_results"]
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]

        st.write(f"✅ Successful: {len(successful)} | ❌ Failed: {len(failed)}")

        # Display each result in expandable sections
        for idx, result in enumerate(results):
            filename = result.get("_filename", f"Image {idx + 1}")

            if "error" in result:
                with st.expander(f"❌ {filename} - Error", expanded=False):
                    st.error(result["error"])
            else:
                artifact_name = result.get("name", "Unknown")
                with st.expander(f"✅ {filename} → {artifact_name}", expanded=False):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        if result.get("_image_data"):
                            st.image(result["_image_data"], use_container_width=True)

                    with col2:
                        st.write(f"**Name:** {result.get('name', 'Unknown')}")
                        st.write(f"**Description:** {result.get('description', 'N/A')}")
                        st.write(f"**Confidence:** {result.get('confidence', 0):.1%}")
                        tags = result.get("tags", [])
                        if tags:
                            st.write(f"**Tags:** {', '.join(tags)}")

                    # Save button for this artifact
                    if st.button(f"💾 Save to Archive", key=f"save_batch_{idx}"):
                        try:
                            tag_list = result.get("tags", [])
                            if isinstance(tag_list, str):
                                tag_list = [
                                    t.strip() for t in tag_list.split(",") if t.strip()
                                ]

                            save_result = create_artifact(
                                name=result.get("name", "Unknown"),
                                description=result.get("description", ""),
                                tags=tag_list,
                                tier=tier,
                                image_data=result["_image_data"],
                                form_data=None,
                                model_3d_data=result.get("_model_3d_data"),
                                model_3d_format=result.get("_model_3d_format"),
                                uploaded_by=st.session_state.get("username"),
                            )
                            st.success(f"✅ Saved! ID: {save_result.get('id')}")
                        except Exception as e:
                            st.error(f"❌ Failed to save: {str(e)}")

        # Save all button
        st.markdown("---")
        if st.button("💾 Save All Successful to Archive", use_container_width=True):
            saved_count = 0
            for idx, result in enumerate(successful):
                try:
                    tag_list = result.get("tags", [])
                    if isinstance(tag_list, str):
                        tag_list = [t.strip() for t in tag_list.split(",") if t.strip()]

                    create_artifact(
                        name=result.get("name", "Unknown"),
                        description=result.get("description", ""),
                        tags=tag_list,
                        tier=tier,
                        image_data=result["_image_data"],
                        form_data=None,
                        model_3d_data=result.get("_model_3d_data"),
                        model_3d_format=result.get("_model_3d_format"),
                        uploaded_by=st.session_state.get("username"),
                    )
                    saved_count += 1
                except Exception as e:
                    st.warning(f"Failed to save {result.get('_filename')}: {str(e)}")

            st.success(f"✅ Saved {saved_count} artifacts to archive!")
            st.balloons()

            # Clear batch results
            del st.session_state["batch_results"]
            st.rerun()


def _make_square_thumbnail_b64(image_b64: str, size: int = 200) -> str:
    """Convert a base64 image to a square thumbnail."""
    try:
        if image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes))

        # Make square
        side = min(image.size)
        left = (image.width - side) / 2
        top = (image.height - side) / 2
        image = image.crop((left, top, left + side, top + side))

        # Resize
        image.thumbnail((size, size))

        # Convert back to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logger.error(f"Thumbnail creation failed: {str(e)}")
        return image_b64


def get_query_params():
    """Helper function to get query params - always returns dict with string values"""
    try:
        params = dict(st.query_params)
    except Exception:
        params = st.experimental_get_query_params()

    # Normalize all values to strings
    normalized = {}
    for k, v in params.items():
        if isinstance(v, list):
            normalized[k] = v[0] if v else ""
        else:
            normalized[k] = str(v) if v is not None else ""
    return normalized


def archive_page():
    """Archive page to view saved artifacts in a gallery."""
    st.header("🏛️ Artifact Archive")

    # Get query parameters
    qp = get_query_params()
    search_query = qp.get("q", "").strip()

    try:
        # Fetch artifacts
        if search_query:
            st.subheader(f"Search Results: {search_query}")
            artifacts = search_artifacts(search_query)
        else:
            artifacts = get_artifacts()

        if not artifacts:
            st.info("📭 No artifacts found. Start by uploading one!")
            return

        # Display gallery statistics
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("Total Artifacts", len(artifacts))
        with col_stats2:
            st.metric("Search Results", len(artifacts) if search_query else "All")

        # Display artifacts in a grid (3 columns) - names only
        st.markdown("---")
        for i in range(0, len(artifacts), 3):
            cols = st.columns(3, gap="medium")

            for j, artifact in enumerate(artifacts[i : i + 3]):
                with cols[j]:
                    # Display image
                    image_url = artifact.get("thumbnail") or artifact.get("image_data")
                    if image_url:
                        st.image(
                            image_url,
                            use_container_width=True,
                        )
                    else:
                        st.warning("No image")

                    # Display only artifact name
                    st.markdown(f"### {artifact.get('name', 'Unknown')}")

                    # View details button - opens popup
                    if st.button(
                        "👁️ View Details",
                        key=f"view_{artifact['id']}",
                        use_container_width=True,
                    ):
                        st.session_state["popup_artifact_id"] = artifact["id"]

        st.markdown("---")

    except Exception as e:
        st.error(f"❌ Failed to load artifacts: {str(e)}")
        logger.exception("Error in archive_page")

    # Handle popup display
    if "popup_artifact_id" in st.session_state:
        artifact_id = st.session_state["popup_artifact_id"]

        # Initialize edit mode state
        if f"edit_mode_{artifact_id}" not in st.session_state:
            st.session_state[f"edit_mode_{artifact_id}"] = False

        try:
            artifact = get_artifact(artifact_id)
            if artifact:
                # Display popup modal
                with st.container(border=True):
                    popup_col1, popup_col2, popup_col3 = st.columns([0.8, 0.1, 0.1])

                    # Edit button
                    with popup_col2:
                        if st.session_state.get("role") in ["admin", "lab", "onsite"]:
                            if st.button(
                                "✏️"
                                if not st.session_state[f"edit_mode_{artifact_id}"]
                                else "💾",
                                key=f"edit_{artifact_id}",
                                help="Edit"
                                if not st.session_state[f"edit_mode_{artifact_id}"]
                                else "Save",
                            ):
                                if st.session_state[f"edit_mode_{artifact_id}"]:
                                    # Save changes
                                    updated_data = {}

                                    # Only add fields that have values
                                    name_val = st.session_state.get(
                                        f"edit_name_{artifact_id}"
                                    )
                                    if name_val:
                                        updated_data["name"] = name_val

                                    desc_val = st.session_state.get(
                                        f"edit_desc_{artifact_id}"
                                    )
                                    if desc_val is not None:  # Allow empty string
                                        updated_data["description"] = desc_val

                                    tags_val = st.session_state.get(
                                        f"edit_tags_{artifact_id}"
                                    )
                                    if tags_val is not None:  # Allow empty string
                                        updated_data["tags"] = tags_val

                                    # Get form data updates
                                    form_updates = {}

                                    # Only add form fields that exist in session state
                                    if f"edit_length_{artifact_id}" in st.session_state:
                                        val = st.session_state[
                                            f"edit_length_{artifact_id}"
                                        ]
                                        if val and val > 0:
                                            form_updates["length"] = val

                                    if f"edit_width_{artifact_id}" in st.session_state:
                                        val = st.session_state[
                                            f"edit_width_{artifact_id}"
                                        ]
                                        if val and val > 0:
                                            form_updates["width"] = val

                                    if (
                                        f"edit_thickness_{artifact_id}"
                                        in st.session_state
                                    ):
                                        val = st.session_state[
                                            f"edit_thickness_{artifact_id}"
                                        ]
                                        if val and val > 0:
                                            form_updates["thickness"] = val

                                    if f"edit_weight_{artifact_id}" in st.session_state:
                                        val = st.session_state[
                                            f"edit_weight_{artifact_id}"
                                        ]
                                        if val and val > 0:
                                            form_updates["weight"] = val

                                    if f"edit_color_{artifact_id}" in st.session_state:
                                        val = st.session_state[
                                            f"edit_color_{artifact_id}"
                                        ]
                                        if val:
                                            form_updates["color"] = val

                                    if (
                                        f"edit_location_{artifact_id}"
                                        in st.session_state
                                    ):
                                        val = st.session_state[
                                            f"edit_location_{artifact_id}"
                                        ]
                                        if val:
                                            form_updates["location"] = val

                                    if (
                                        f"edit_phys_desc_{artifact_id}"
                                        in st.session_state
                                    ):
                                        val = st.session_state[
                                            f"edit_phys_desc_{artifact_id}"
                                        ]
                                        if val:
                                            form_updates["description"] = val

                                    if (
                                        f"edit_artifact_name_{artifact_id}"
                                        in st.session_state
                                    ):
                                        val = st.session_state[
                                            f"edit_artifact_name_{artifact_id}"
                                        ]
                                        if val:
                                            form_updates["artifact_name"] = val

                                    if form_updates:
                                        updated_data["form_data"] = form_updates

                                    if updated_data:
                                        try:
                                            logger.info(
                                                f"Saving artifact {artifact_id} with data: {updated_data}"
                                            )
                                            logger.info(
                                                f"Session state keys: {[k for k in st.session_state.keys() if str(artifact_id) in k]}"
                                            )
                                            response = api_request(
                                                "PUT",
                                                f"api/artifacts/{artifact_id}",
                                                data=updated_data,
                                            )
                                            logger.info(f"API response: {response}")
                                            st.success("✅ Changes saved!")
                                            st.session_state[
                                                f"edit_mode_{artifact_id}"
                                            ] = False
                                            st.rerun()
                                        except Exception as e:
                                            error_msg = str(e)
                                            logger.error(
                                                f"Failed to save artifact: {error_msg}"
                                            )
                                            logger.error(
                                                f"Update data was: {updated_data}"
                                            )
                                            st.error(f"Failed to save: {error_msg}")
                                    else:
                                        st.warning("No changes to save")
                                else:
                                    # Enter edit mode
                                    st.session_state[f"edit_mode_{artifact_id}"] = True
                                    st.rerun()

                    # Close button
                    with popup_col3:
                        if st.button("✕", key=f"close_{artifact_id}"):
                            del st.session_state["popup_artifact_id"]
                            if f"edit_mode_{artifact_id}" in st.session_state:
                                del st.session_state[f"edit_mode_{artifact_id}"]
                            st.rerun()

                    with popup_col1:
                        if st.session_state[f"edit_mode_{artifact_id}"]:
                            # Edit mode - show input field
                            edited_name = st.text_input(
                                "Name",
                                value=artifact.get("name", "Unknown"),
                                key=f"edit_name_{artifact_id}",
                            )
                        else:
                            st.markdown(f"## {artifact.get('name', 'Unknown')}")

                    # Create two columns for layout in popup
                    detail_left, detail_right = st.columns([1, 2])

                    # Left: Image and 3D Model
                    with detail_left:
                        image_url = artifact.get("image_data") or artifact.get(
                            "thumbnail"
                        )
                        if image_url:
                            st.image(image_url, use_container_width=True)
                        else:
                            st.warning("No image available")

                        # 3D Model viewer if available
                        if artifact.get("has_3d_model") and artifact.get(
                            "model_3d_data"
                        ):
                            st.markdown("---")
                            st.markdown("**🎯 3D Model**")
                            model_format = artifact.get("model_3d_format", "obj")

                            if HAS_3D_VIEWER and model_format == "obj":
                                # Decode and display 3D model
                                try:
                                    model_data_url = artifact["model_3d_data"]
                                    if "," in model_data_url:
                                        model_b64 = model_data_url.split(",")[1]
                                    else:
                                        model_b64 = model_data_url
                                    model_bytes = base64.b64decode(model_b64)

                                    # Save to temp file for viewer
                                    with tempfile.NamedTemporaryFile(
                                        delete=False, suffix=".obj"
                                    ) as tmp_file:
                                        tmp_file.write(model_bytes)
                                        tmp_path = tmp_file.name

                                    with st.expander("🔍 View 3D Model", expanded=True):
                                        viewer = Model3DViewer()
                                        viewer.render_3d_model(tmp_path, height=350)

                                    # Clean up temp file
                                    try:
                                        os.unlink(tmp_path)
                                    except:
                                        pass
                                except Exception as e:
                                    st.warning(f"Could not render 3D model: {e}")
                                    logger.error(f"3D model render error: {e}")
                            else:
                                # Provide download button for non-OBJ formats or when viewer unavailable
                                st.info(
                                    f"3D model available ({model_format.upper()} format)"
                                )
                                model_data_url = artifact["model_3d_data"]
                                if "," in model_data_url:
                                    model_b64 = model_data_url.split(",")[1]
                                else:
                                    model_b64 = model_data_url
                                model_bytes = base64.b64decode(model_b64)
                                obj_text = model_bytes.decode('utf-8')
                                 # make temporary .obj file
                                tmp_path = ""
                                with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
                                    tmp.write(obj_text.encode("utf-8"))
                                    tmp_path = tmp.name

                                html_output = obj2html.obj2html(
                                    tmp_path, html_elements_only=True
                                )
                                st.components.v1.html(
                                    html_output,width='stretch',height='stretch',  scrolling=True
                                )
                                st.download_button(
                                    label=f"📥 Download 3D Model (.{model_format})",
                                    data=model_bytes,
                                    file_name=f"artifact_{artifact_id}.{model_format}",
                                    mime=f"model/{model_format}",
                                    key=f"download_3d_{artifact_id}",
                                )

                    # Right: Details
                    with detail_right:
                        edit_mode = st.session_state[f"edit_mode_{artifact_id}"]

                        # Core information
                        st.markdown("**Basic Information:**")
                        st.write(f"- **ID:** {artifact.get('id')}")
                        st.write(f"- **Tier:** {artifact.get('tier', 'N/A')}")
                        st.write(
                            f"- **Uploaded:** {artifact.get('uploaded_at', 'N/A')}"
                        )

                        # Description
                        desc = artifact.get("description", "")
                        if edit_mode:
                            st.markdown("**Description:**")
                            edited_desc = st.text_area(
                                "",
                                value=desc,
                                height=100,
                                key=f"edit_desc_{artifact_id}",
                                label_visibility="collapsed",
                            )
                        elif desc:
                            st.markdown("**Description:**")
                            st.write(desc)

                        # Tags
                        tags = artifact.get("tags", [])
                        if isinstance(tags, str):
                            tags = [t.strip() for t in tags.split(",")]

                        if edit_mode:
                            st.markdown("**Tags:**")
                            tags_str = ", ".join(tags) if tags else ""
                            edited_tags = st.text_input(
                                "",
                                value=tags_str,
                                key=f"edit_tags_{artifact_id}",
                                label_visibility="collapsed",
                                placeholder="Enter comma-separated tags",
                            )
                        elif tags:
                            st.markdown("**Tags:**")
                            for tag in tags:
                                st.write(f"🏷️ {tag}")

                        # Verification status
                        status = artifact.get("verification_status", "pending")
                        if status == "verified":
                            st.success(f"✅ Verified")
                        elif status == "rejected":
                            st.error("❌ Rejected")
                        else:
                            st.info("⏳ Pending verification")

                    # Form data if available - full width
                    form_data = artifact.get("form_data")
                    if form_data:
                        st.markdown("---")
                        st.markdown("**📐 Physical Measurements & Details**")

                        # Parse form data if it's a JSON string
                        if isinstance(form_data, str):
                            import json

                            try:
                                form_data = json.loads(form_data)
                            except (json.JSONDecodeError, TypeError):
                                form_data = {}

                        if isinstance(form_data, dict):
                            edit_mode = st.session_state[f"edit_mode_{artifact_id}"]

                            # Display measurements in columns
                            meas_col1, meas_col2, meas_col3 = st.columns(3)

                            with meas_col1:
                                if edit_mode:
                                    st.number_input(
                                        "Length (cm)",
                                        value=float(form_data.get("length", 0) or 0),
                                        key=f"edit_length_{artifact_id}",
                                        min_value=0.0,
                                        step=0.1,
                                    )
                                    st.number_input(
                                        "Width (cm)",
                                        value=float(form_data.get("width", 0) or 0),
                                        key=f"edit_width_{artifact_id}",
                                        min_value=0.0,
                                        step=0.1,
                                    )
                                else:
                                    if form_data.get("length"):
                                        st.metric("Length", f"{form_data['length']} cm")
                                    if form_data.get("width"):
                                        st.metric("Width", f"{form_data['width']} cm")

                            with meas_col2:
                                if edit_mode:
                                    st.number_input(
                                        "Thickness (cm)",
                                        value=float(form_data.get("thickness", 0) or 0),
                                        key=f"edit_thickness_{artifact_id}",
                                        min_value=0.0,
                                        step=0.1,
                                    )
                                    st.number_input(
                                        "Weight (g)",
                                        value=float(form_data.get("weight", 0) or 0),
                                        key=f"edit_weight_{artifact_id}",
                                        min_value=0.0,
                                        step=0.1,
                                    )
                                else:
                                    if form_data.get("thickness"):
                                        st.metric(
                                            "Thickness", f"{form_data['thickness']} cm"
                                        )
                                    if form_data.get("weight"):
                                        st.metric("Weight", f"{form_data['weight']} g")

                            with meas_col3:
                                if edit_mode:
                                    st.text_input(
                                        "Color",
                                        value=form_data.get("color", ""),
                                        key=f"edit_color_{artifact_id}",
                                    )
                                    st.text_input(
                                        "Location",
                                        value=form_data.get("location", ""),
                                        key=f"edit_location_{artifact_id}",
                                    )
                                else:
                                    if form_data.get("color"):
                                        st.write("**Color:** ", form_data["color"])
                                    if form_data.get("location"):
                                        st.write(
                                            f"**Location:** {form_data['location']}"
                                        )

                            # Display description if available
                            if edit_mode:
                                st.markdown("**Physical Description:**")
                                st.text_area(
                                    "",
                                    value=form_data.get("description", ""),
                                    key=f"edit_phys_desc_{artifact_id}",
                                    height=100,
                                    label_visibility="collapsed",
                                )
                            elif form_data.get("description"):
                                st.markdown("**Physical Description:**")
                                st.write(form_data["description"])

                            # Display artifact name from form if available
                            if edit_mode:
                                st.markdown("**Artifact Name (from form):**")
                                st.text_input(
                                    "",
                                    value=form_data.get("artifact_name", ""),
                                    key=f"edit_artifact_name_{artifact_id}",
                                    label_visibility="collapsed",
                                )
                            elif form_data.get("artifact_name"):
                                st.markdown(
                                    f"**Artifact Name (from form):** {form_data['artifact_name']}"
                                )

                            # Display tags from form if available
                            form_tags = form_data.get("tags", [])
                            if form_tags:
                                if isinstance(form_tags, str):
                                    form_tags = [
                                        t.strip()
                                        for t in form_tags.split(",")
                                        if t.strip()
                                    ]
                                st.markdown("**Form Tags:**")
                                tag_cols = st.columns(
                                    len(form_tags) if form_tags else 1
                                )
                                for idx, tag in enumerate(form_tags):
                                    with tag_cols[idx]:
                                        st.write(f"🏷️ {tag}")
                        else:
                            st.json(form_data)

                # Approval/Rejection buttons for admin users (at the bottom)
                if st.session_state.get("role") in ["admin", "lab", "onsite"]:
                    st.markdown("---")
                    st.markdown("**🔐 Verification Actions**")

                    # Show uploader info if available
                    uploaded_by = artifact.get("uploaded_by")
                    if uploaded_by:
                        st.info(f"📤 Uploaded by: **{uploaded_by}**")

                    # Reason input (required)
                    verification_reason = st.text_area(
                        "Reason for decision (required)*",
                        placeholder="Please provide a detailed reason for your decision. This will be sent to the uploader.",
                        key=f"reason_{artifact_id}",
                        height=100,
                    )

                    col_approve, col_reject = st.columns(2)
                    with col_approve:
                        if st.button(
                            "✅ Approve",
                            key=f"approve_{artifact_id}",
                            use_container_width=True,
                        ):
                            if (
                                not verification_reason
                                or not verification_reason.strip()
                            ):
                                st.error("❌ Please provide a reason for approval")
                            else:
                                try:
                                    result = api_request(
                                        "POST",
                                        f"api/artifacts/{artifact_id}/verify",
                                        data={
                                            "verification_status": "verified",
                                            "reason": verification_reason.strip(),
                                            "verified_by": st.session_state.get(
                                                "username", "unknown"
                                            ),
                                        },
                                    )
                                    st.success("✅ Artifact approved!")
                                    if result.get("email_sent"):
                                        st.info(
                                            "📧 Notification sent to uploader on Whatsapp or Email"
                                        )
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to approve artifact: {str(e)}")
                    with col_reject:
                        if st.button(
                            "❌ Reject",
                            key=f"reject_{artifact_id}",
                            use_container_width=True,
                        ):
                            if (
                                not verification_reason
                                or not verification_reason.strip()
                            ):
                                st.error("❌ Please provide a reason for rejection")
                            else:
                                try:
                                    result = api_request(
                                        "POST",
                                        f"api/artifacts/{artifact_id}/verify",
                                        data={
                                            "verification_status": "rejected",
                                            "reason": verification_reason.strip(),
                                            "verified_by": st.session_state.get(
                                                "username", "unknown"
                                            ),
                                        },
                                    )
                                    st.success("❌ Artifact rejected!")
                                    if result.get("email_sent"):
                                        st.info(
                                            "📧 Notification sent to uploader on Whatsapp or Email"
                                        )
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to reject artifact: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error loading artifact: {str(e)}")
            logger.exception("Error loading artifact details")


def search_page():
    """Search page for advanced artifact search."""
    st.header("🔍 Advanced Search")

    search_query = st.text_input("Enter search term")

    if search_query:
        try:
            results = search_artifacts(search_query)

            if results:
                st.success(f"Found {len(results)} artifacts")

                for artifact in results:
                    with st.expander(
                        f"{artifact.get('name')} (ID: {artifact.get('id')})"
                    ):
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            image_url = artifact.get("thumbnail") or artifact.get(
                                "image_data"
                            )
                            if image_url:
                                st.image(image_url, use_container_width=True)

                        with col2:
                            st.write(
                                f"**Description:** {artifact.get('description', 'N/A')}"
                            )
                            artifact_tags = artifact.get("tags", [])
                            if isinstance(artifact_tags, str):
                                artifact_tags = [
                                    t.strip()
                                    for t in artifact_tags.split(",")
                                    if t.strip()
                                ]
                            elif isinstance(artifact_tags, list):
                                artifact_tags = [
                                    str(t).strip() for t in artifact_tags if t
                                ]
                            st.write(f"**Tags:** {', '.join(artifact_tags)}")
                            st.write(f"**Tier:** {artifact.get('tier', 'N/A')}")
            else:
                st.info("No artifacts found matching your search")
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
