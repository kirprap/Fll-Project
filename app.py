import streamlit as st
from PIL import Image
from datetime import datetime
from io import BytesIO
from ai_analyzer import AIAnalyzer

# Assume your database functions exist:
from database import (
    save_artifact,
    get_all_artifacts,
    search_artifacts,
    get_artifact_by_id,
    update_artifact_verification,
    update_artifact_profile,
    get_artifacts_by_verification_status,
)


# ------------------------------
# Helpers
# ------------------------------
def pil_to_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


# ------------------------------
# Streamlit
# ------------------------------
analyzer = AIAnalyzer()

if "search_history" not in st.session_state:
    st.session_state.search_history = []

st.set_page_config(
    page_title="Archaeological Artifact Identifier", page_icon="🏺", layout="wide"
)

st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose model", ["vit", "clip"])

st.title("🏺 Archaeological Artifact Identifier")
picture = st.camera_input("Take a picture")

# If a picture is taken, you can process or save it
if picture:
    st.image(picture, caption="Captured Image", use_column_width=True)
    # To save the picture:
    # with open("captured_image.jpg", "wb") as f:
    #     f.write(picture.getvalue())
    # st.success("Image saved!")

uploaded_file = st.file_uploader("Upload artifact image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Artifact", use_column_width=True)

    if st.button("Analyze Artifact"):
        with st.spinner("Running AI analysis..."):
            result = analyzer.analyze_image(image, model_choice=model_choice)

            # Save artifact

            try:
                image_bytes = pil_to_bytes(image)

                artifact_id = save_artifact(result, image_bytes)

                st.success(f"Saved artifact with ID: {artifact_id}!")

            except Exception as e:
                st.error(f"Failed to save artifact: {e}")

            # Add to search history
            st.session_state.search_history.append({"image": image, "result": result})

            # Show result
            st.subheader("Analysis Result")
            if model_choice == "vit":
                st.write(f"Name: {result['name']}")
                st.write(f"Confidence: {result['confidence'] * 100:.1f}%")
            else:
                st.write("CLIP embedding computed (used for similarity search).")

# ------------------------------
# Similarity Search
# ------------------------------
st.subheader("🔎 Similarity Search")

if st.button("Run Similarity Search"):
    # Make sure we have a result from analysis
    if "result" in locals() or "result" in globals():
        all_arts = get_all_artifacts(limit=50)

        query_embedding = result.get("embedding")

        if query_embedding is not None and all_arts:
            # Add embeddings to artifacts for similarity search
            for art in all_arts:
                # In a real app, you'd store embeddings in the database
                # For now, we'll skip artifacts without embeddings
                pass
            sim_res = analyzer.similarity_search(query_embedding, all_arts)

            if sim_res:
                st.write(
                    f"Closest Match: {sim_res['closest_match']} ({sim_res['similarity_score'] * 100:.0f}% similarity)"
                )

                st.write("Alternative Matches:")

                for alt in sim_res.get("alternative_matches", []):
                    st.write(
                        f"• {alt['artifact']['name']} ({alt['score'] * 100:.0f}% similarity)"
                    )
    else:
        st.warning("Please analyze an artifact first before running similarity search.")


# ------------------------------
# Search History
# ------------------------------
if st.session_state.search_history:
    st.subheader("🕐 Search History")
    for entry in reversed(st.session_state.search_history):
        st.image(entry["image"], width=150)
        st.write(entry["result"])
