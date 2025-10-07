import streamlit as st

def load_google_fonts():
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Montserrat:wght@600&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

def apply_custom_css():
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        h1, h2 {
            font-family: 'Montserrat', sans-serif;
        }
        .metric-card {
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 10px;
        }
        .metric-title {
            font-size: 14px;
            color: #666;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
        }
        .metric-change {
            font-size: 14px;
            color: green;
        }
        .framework-section {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .framework-title {
            color: #1f77b4;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
