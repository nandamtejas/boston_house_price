import streamlit as st

if __name__ == "__main__":
    st.set_page_config(
        page_title="About",
        page_icon="📝"
    )
    st.title("Boston House Price")
    st.markdown("""
    The Boston Housing Dataset is a derived from information collected by the U.S.
    Census Service concerning housing in the area of Boston MA.

    ![image](https://user-images.githubusercontent.com/65908099/204079041-efd4131d-27f7-48df-9cf9-e389e263712c.png)
    """)