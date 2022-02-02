import streamlit as st
import pandas as pd

path = st.text_input('CSV file path')
if path:
    df = pd.read_csv(path)
    df
