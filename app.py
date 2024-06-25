import streamlit as st
from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import io as io_module
from PIL import Image

def main():    
  st.title("Image Upload and Display")
  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    image = io.imread(uploaded_file)
    rows = image.shape[0]
    cols = image.shape[1]
    image = image.reshape(rows*cols, 3)
    kmeans = KMeans(n_clusters=16)
    kmeans.fit(image)
    centroids=kmeans.cluster_centers_
    labels=kmeans.labels_
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)
    compressed_image = compressed_image.reshape(rows, cols, 3)
    st.image(compressed_image)
    
    pil_image = Image.fromarray(compressed_image)
    buf = io_module.BytesIO()
    pil_image.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    
    size_kb = len(byte_im) / 1024
    st.write(f"Size of the compressed image: {size_kb:.2f} KB")
    
    st.download_button(label="Download Processed Image", data=byte_im, file_name="processed_image.jpg", mime="image/jpeg")

if __name__ == '__main__':
  main()

    