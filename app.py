import os
import re
import json
import tempfile
import textwrap
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# Fix indented HTML rendering in Streamlit
_original_markdown = st.markdown

def fixed_markdown(body, *args, **kwargs):
    if isinstance(body, str):
        body = textwrap.dedent(body)
    return _original_markdown(body, *args, **kwargs)

st.markdown = fixed_markdown


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
