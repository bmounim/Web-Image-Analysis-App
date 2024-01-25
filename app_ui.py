# app_ui.py

import streamlit as st

def load_css():
    """
    Loads the CSS for the Streamlit app.
    """
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def render_navbar():
    """
    Renders the navigation bar for the app.
    """
    navbar_html = """
    <nav class="navbar">
        <a href="#home">Home</a>
        <a href="#create-account">Create Account</a>
        <a href="#login">Log In</a>
        <a href="#pricing">Pricing</a>
        <a href="#about">About</a>
    </nav>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)

def render_header():
    """
    Renders the header section of the app.
    """
    header_html = """
        <div class="app-header">
            <h1 class="app-title">Webpage Image Analysis Tool</h1>
            <p class="app-description">Capture and analyze website screenshots with AI-powered technology.</p>
        </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_input_section():
    """
    Renders the input section where users can input data.
    """
    st.markdown('<div class="input-section"><h2>Start Analysis</h2></div>', unsafe_allow_html=True)

    # List of countries for selection
    countries = ["Germany"]
    selected_country = st.selectbox("Choose a country:", countries)

    url_input = st.text_input("Enter the URL of the website:", "")

    return url_input, selected_country

def render_about_section():
    """
    Renders the about section of the app.
    """
    about_html = """
        <div class="info-section">
            <h2>About the Tool</h2>
            <p>This tool utilizes state-of-the-art AI algorithms to analyze website images and generate insights.</p>
        </div>
    """
    st.markdown(about_html, unsafe_allow_html=True)

def render_footer():
    """
    Renders the footer of the app.
    """
    footer_html = '<div class="footer">© 2024 Web Analysis Tool. All rights reserved.</div>'
    st.markdown(footer_html, unsafe_allow_html=True)

def render_download_button(xlsx_data):
    """
    Renders a download button for the XLSX data.
    :param xlsx_data: The XLSX data to be downloaded.
    """
    st.download_button(
        label="Download Results as XLSX",
        data=xlsx_data,
        file_name="analysis_results.xlsx",
        mime="application/vnd.ms-excel"
    )
