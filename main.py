import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings 
warnings.filterwarnings('ignore')

filepath = "\\Boston_"


@st.cache()
def load_dataset(file=filepath):
    _1 = pd.read_csv(file + "Train.csv", index_col=0)
    _2 = pd.read_csv(file + "Test.csv", index_col=0)
    df = pd.concat([_1, _2])
    df.replace(to_replace=0, value=np.NaN, inplace=True)
    
    # columns with null values
    null_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    # remove the columns
    df.drop(columns=null_cols, inplace=True)

    # linearity of `crim`
    df['crim'] = np.log(1+df['crim'])

    # linearity of black
    df['black'] = np.log(1+df['black'])

    #linearity of lstat
    df['lstat'] = np.log(1+df['lstat'])

    # for col in df.columns:
    #     df[col] = np.log(1+df[col])

    return df

@st.cache
def load_observations():
    import json 
    with open("observations.json", "r") as file:
        data = json.loads(file.read())
        st.session_state['observations'] = data
    return st.session_state['observations']




if __name__ == "__main__":

    st.set_page_config(
        page_title="Main Page",
        page_icon="🎈",
        layout="wide"
    )
    st.markdown("# Boston House Prices") 
    st.sidebar.header("Boston House Prices")

    df = load_dataset(filepath)

    rows = st.sidebar.slider('rows', min_value=0, max_value=len(df), value=5, step=5)

    if "dataframe" not in st.session_state:
        st.session_state['dataframe'] = df

    if st.sidebar.checkbox("Show Dataframe", True):
        st.markdown("### Dataframe")
        st.dataframe(df.head(n=rows), use_container_width=True)
    if st.sidebar.checkbox("Show Dataframe Description"):
        st.markdown("### Description", True)
        st.table(df.describe())
    if st.sidebar.checkbox("Show Correlation Table"):
        st.markdown("### Correlation Matrix", True)
        st.dataframe(round(df.corr(), 2), use_container_width=True)
    corr = round(df.corr(), 2)
    s = {col: round(df["medv"].corr(df[col]), 2) for col in df.columns if col != "medv"}
    result = min(s, key=s.get), max(s, key=s.get)
    st.session_state['corr'] = {res: s[res] for res in result}


    st.write("""
        MADE BY TEJAS NANDAM
    """)