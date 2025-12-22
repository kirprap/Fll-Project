import os
import tempfile
import time
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components
from obj2html import obj2html
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import logging

logger = logging.getLogger(__name__)


class Model3DViewer:
    """3D Model Viewer for OBJ files with screenshot capabilities"""
    
    def __init__(self):
        self.driver = None
        
    def render_3d_model(self, obj_file_path: str, height: int = 600) -> bool:
        """
        Render a 3D model using obj2html in Streamlit
        
        Args:
            obj_file_path: Path to the OBJ file
            height: Height of the viewer in pixels
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(obj_file_path):
                st.error(f"Error: The file '{obj_file_path}' was not found.")
                return False
                
            # Convert the OBJ file to an HTML string
            html_string = obj2html(obj_file_path, html_elements_only=True)
            
            # Render the HTML string in Streamlit
            components.html(html_string, height=height, scrolling=True)
            
            # Add download button for the OBJ file
            with open(obj_file_path, "rb") as f:
                st.download_button(
                    label="Download 3D Model",
                    data=f,
                    file_name=os.path.basename(obj_file_path),
                    mime="model/obj"
                )
            
            return True
            
        except Exception as e:
            st.error(f"An error occurred while rendering the 3D model: {e}")
            logger.error(f"3D model rendering error: {e}")
            return False
    
    def setup_selenium_driver(self) -> bool:
        """Setup Selenium Chrome driver for screenshots"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Use webdriver-manager to automatically handle driver installation
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Selenium driver: {e}")
            st.error(f"Failed to setup screenshot driver: {e}")
            return False
    
    def capture_model_screenshots(self, streamlit_url: str = "http://localhost:8501", 
                                output_dir: str = "screenshots") -> list:
        """
        Capture screenshots of the 3D model from different angles
        
        Args:
            streamlit_url: URL of the Streamlit app
            output_dir: Directory to save screenshots
            
        Returns:
            list: Paths to captured screenshots
        """
        if not self.driver:
            if not self.setup_selenium_driver():
                return []
        
        screenshots = []
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Navigate to the Streamlit app
            self.driver.get(streamlit_url)
            time.sleep(3)  # Wait for page to load
            
            # Capture screenshots from different angles/positions
            angles = [
                {"name": "front", "scroll": 0},
                {"name": "side", "scroll": 200},
                {"name": "top", "scroll": 100},
                {"name": "angled", "scroll": 150}
            ]
            
            for angle in angles:
                try:
                    # Scroll to position if needed
                    if angle["scroll"] > 0:
                        self.driver.execute_script(f"window.scrollTo(0, {angle['scroll']});")
                        time.sleep(1)
                    
                    # Capture screenshot
                    screenshot_path = os.path.join(output_dir, f"model_{angle['name']}_{int(time.time())}.png")
                    self.driver.save_screenshot(screenshot_path)
                    screenshots.append(screenshot_path)
                    
                    st.success(f"Captured {angle['name']} view: {screenshot_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to capture {angle['name']} screenshot: {e}")
                    st.warning(f"Failed to capture {angle['name']} view")
            
            return screenshots
            
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            st.error(f"Screenshot capture failed: {e}")
            return []
        
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
    
    def cleanup_driver(self):
        """Clean up Selenium driver"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            finally:
                self.driver = None


def save_uploaded_obj_file(uploaded_file) -> Optional[str]:
    """
    Save an uploaded OBJ file to a temporary location
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Path to saved file or None if failed
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Failed to save uploaded OBJ file: {e}")
        st.error(f"Failed to save 3D model file: {e}")
        return None


def render_3d_model_section():
    """Render a complete 3D model upload and viewer section"""
    st.header("🎯 3D Model Viewer")
    
    # File upload
    uploaded_obj = st.file_uploader(
        "Upload 3D Model (OBJ file)",
        type=["obj"],
        help="Upload a 3D model in OBJ format for viewing and analysis"
    )
    
    if uploaded_obj is not None:
        # Save uploaded file
        temp_file_path = save_uploaded_obj_file(uploaded_obj)

        if temp_file_path:
            try:
                st.success(f"3D model uploaded: {uploaded_obj.name}")
                
                # Create viewer instance
                viewer = Model3DViewer()
                
                # Viewer options
                col1, col2 = st.columns(2)
                with col1:
                    viewer_height = st.slider("Viewer Height", 400, 800, 600)
                with col2:
                    enable_screenshots = st.checkbox("Enable Screenshot Analysis")
                
                # Render 3D model
                st.markdown("### 3D Model View")
                if viewer.render_3d_model(temp_file_path, height=viewer_height):
                    st.success("3D model loaded successfully!")
                
                # Screenshot analysis section
                if enable_screenshots:
                    st.markdown("---")
                    st.markdown("### 📸 Screenshot Analysis")
                    
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        screenshot_dir = st.text_input("Screenshot Directory", "screenshots")
                    with col_s2:
                        streamlit_url = st.text_input("Streamlit URL", "http://localhost:8501")
                    
                    if st.button("📸 Capture Screenshots", use_container_width=True):
                        with st.spinner("Capturing screenshots from different angles..."):
                            screenshots = viewer.capture_model_screenshots(
                                streamlit_url=streamlit_url,
                                output_dir=screenshot_dir
                            )
                            
                            if screenshots:
                                st.success(f"Captured {len(screenshots)} screenshots")
                                
                                # Display screenshots
                                st.markdown("#### Captured Screenshots:")
                                for i, screenshot_path in enumerate(screenshots):
                                    if os.path.exists(screenshot_path):
                                        st.image(screenshot_path, caption=f"View {i+1}", use_container_width=True)
                            else:
                                st.error("Failed to capture screenshots")
                
            except Exception as e:
                st.error(f"Error processing 3D model: {e}")
                logger.error(f"3D model processing error: {e}")
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass


if __name__ == "__main__":
    # Test the 3D viewer
    render_3d_model_section()
