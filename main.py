import streamlit as st

# Title
st.title("Welcome to My Streamlit App")

# Text input
name = st.text_input("Enter your name:")

# Button and output
if st.button("Say Hello"):
    st.success(f"Hello, {name}!")