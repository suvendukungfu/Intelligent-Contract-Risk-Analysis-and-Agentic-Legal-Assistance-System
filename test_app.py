import streamlit as st
st.title("Streamlit Test Page")
st.write("If you can see this, Streamlit is working!")
uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    st.write(f"File uploaded: {uploaded_file.name}")
    st.write(f"File size: {len(uploaded_file.getvalue())} bytes")
