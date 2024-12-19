import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import glob
from io import BytesIO


st.set_page_config(
        page_title="House Transformation",
)

@st.cache_data
def load_palette_images():
    palette_files = glob.glob("./palettes/*.png")
    palettes = {os.path.splitext(os.path.basename(f))[0]: Image.open(f) for f in palette_files}
    return palettes

@st.cache_data
def load_roof_palette_images():
    roof_palette_files = glob.glob("./roofs/*.png")
    roof_palettes = {os.path.splitext(os.path.basename(f))[0]: Image.open(f) for f in roof_palette_files}
    return roof_palettes

def main():
    st.title("LTI: House Segmentation and Blending App")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG format)", type=["jpg", "png"])
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load segmentation mask by filename
        filename_no_ext = os.path.splitext(uploaded_file.name)[0]
        mask_path = f"./masks/{filename_no_ext}.png"
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = np.array(mask)[..., 0]

            # Create dropdown list for segmented areas (excluding background)
            unique_values = np.unique(mask)
            unique_values = unique_values[unique_values != 0]  # Exclude background (value 0)
            segment_labels = [f"{value} - {'wall' if value == 1 else 'roof' if value == 2 else 'bricks' if value == 3 else 'area'}" for value in unique_values]
            selected_area_value = st.selectbox("Select a segmented area", unique_values, format_func=lambda x: segment_labels[unique_values.tolist().index(x)])

            # Draw segmentation area on the picture
            segmentation_area = np.repeat((mask == selected_area_value).astype(np.uint8)[..., np.newaxis], 3, axis=2)
            segmentation_area = np.multiply(segmentation_area, np.array([255,192,203])).astype(np.uint8)
            segmentation_overlay = cv2.addWeighted(st.session_state.get('blended_image', image), 0.5, segmentation_area, 0.5, 0)
            st.image(segmentation_overlay, caption='Highlighted Segmentation Area', use_column_width=True)

            # Load color palettes based on selected area
            if selected_area_value == 2:  # Roof area
                palettes = load_roof_palette_images()
            else:
                palettes = load_palette_images()
            palette_names = list(palettes.keys())
            selected_palette_name = st.selectbox("Select a color palette", palette_names)
            selected_palette = np.array(palettes[selected_palette_name])[..., :3]

            # Show palette preview
            st.image(selected_palette, caption='Selected Color Palette', use_column_width=False, width=150)

            # Slider for blending ratio
            beta = st.slider("Blending Ratio", min_value=0.0, max_value=1.0, step=0.1, value=0.5)

            # Apply button
            if st.button("Apply"):
                blended_image = blend_images_with_mask(st.session_state.get('blended_image', image), (mask == selected_area_value).astype(np.uint8), selected_palette, beta=beta)
                st.session_state['blended_image'] = blended_image
                st.image(blended_image, caption='Blended Image', use_column_width=True)

            # Download blended image
            if 'blended_image' in st.session_state:
                final_image = Image.fromarray(st.session_state['blended_image'])
                buffered = BytesIO()
                final_image.save(buffered, format="PNG")
                st.download_button(
                    label="Download Blended Image",
                    data=buffered.getvalue(),
                    file_name="blended_image.png",
                    mime="image/png"
                )

def blend_images_with_mask(image, mask, second_image, beta=0.5):
    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    # Resize second image to match the dimensions of the first image
    second_image_resized = cv2.resize(second_image, (image.shape[1], image.shape[0]))
    
    # Blend image with the second image using mask
    blended_image = image.copy()
    blended_image[mask] = cv2.addWeighted(image[mask], 1-beta, second_image_resized[mask], beta, 0)
    
    return blended_image

if __name__ == "__main__":
    main()