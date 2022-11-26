import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components

import sys
sys.setrecursionlimit(10000)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Heat Map",
    page_icon="H",
    layout="wide"
)

from main import load_dataset, load_observations


@st.cache
def get_dataset():
    try:
        return st.session_state['dataframe']
    except KeyError as err:
        return load_dataset()

@st.cache
def get_observations():
    try: 
        return st.session_state['observations']
    except KeyError as err:
        return load_observations()


df = get_dataset()
observations = get_observations()

@st.cache
def get_sample_dataframe(samples=10):
    return df.sample(samples)

def get_heat_map(data=df, annot=False):
    corr = round(data.corr(), 2)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=annot, ax=ax)
    return fig


st.sidebar.header("Heat Map")
st.markdown("# Boston House Prices")

@st.cache
def get_correlation_values(data=round(df, 2)):
    s = {col: round(data["medv"].corr(data[col]), 2) for col in df.columns if col != "medv"}
    result = min(s, key=s.get), max(s, key=s.get)
    return {res: s[res] for res in result}


if __name__ == "__main__":

    col1, col2 = st.columns(2)


    samples = st.sidebar.slider("samples", min_value=10, max_value=len(df), value=120, step=10)
    df_samples = get_sample_dataframe(samples)

    annot = st.sidebar.checkbox("annot", True)


    with col1:
        
        fig = get_heat_map(df_samples, annot)
        st.pyplot(fig)
    
    with col2:
        with st.container():
            st.write("## Correlation")
            
        with st.expander("Observation"):
            d = get_correlation_values(df_samples)
            st.markdown(f"""
                The plot is called `Heatmap` which shows the correlation between each column. 
                The value of the correlation is ranging from `(-1, 1)`. 
                The highest correlation is always `1`.

                - If the value is closer to `1`, it means there is strong positive correlation between two variables.
                - If the value is closer to `-1`, it means there is strong negative correlation between two variables.

                ### Observations

                - `RM` has a strong positive correlation with MEDV `({d.get('rm', 0)})`
                -  `LSTAT` has a high negative correlation with MEDV `({d.get('lstat', 0)})`
            """)
        

    #fig_to_html = mpld3.fig_to_html(fig)
    #components.html(fig_to_html, height=10000, width=55350, scrolling=True)