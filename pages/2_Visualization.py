import streamlit as st 
import seaborn as sns
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components

import sys
sys.setrecursionlimit(10000)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Visualization",
    page_icon="📈",
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

@st.experimental_memo
def pairplot(kind, data=df, shade=False):
    if kind == "kde":
        plot = sns.pairplot(data=data, kind=kind, diag_kws={"shade":shade})
    else:
        plot = sns.pairplot(data=data, kind=kind)
    return plot

def jointplot(xd, kind, data=df, shade=False):
    if kind == "kde":    
        plot = sns.jointplot(x=xd, y="medv", data=data, kind=kind, marginal_kws={'shade': shade}, ratio=2, space=0)
    else:
        plot = sns.jointplot(x=xd, y="medv", data=data, kind=kind, ratio=2, space=0)
    plot.fig.suptitle(observations[xd]['title'])
    return plot


def lineplot(xd, _ax, data=df):
    sns.lineplot(x=xd, y="medv", data=data, ax=_ax).set(title=observations[xd]['title'])

def scatterplot(xd, _ax, data=df):
    sns.scatterplot(x=xd, y="medv", data=data, ax=_ax).set(title=observations[xd]['title'])

def countplot(xd, _ax, data=df):
    sns.countplot(x=xd, data=data, ax=_ax).set(title=observations[xd]['title'])

def stripplot(xd, _ax, data=df):
    sns.stripplot(x=xd, y="medv", data=data, ax=_ax).set(title=observations[xd]['title'])

def densityplot(xd, _ax, data=df, shade=False):
    sns.kdeplot(x=xd, data=data, ax=_ax, shade=shade).set(title=observations[xd]['title'])

@st.experimental_memo
def get_plot(sd, xd="crim", kind=None, data=df, shade=False):
    fig, ax = plt.subplots()
    if sd == "Joint Plot":
        plot = jointplot(xd=xd, kind=kind, data=data, shade=shade)
        return plot.figure
    elif sd == "Line Plot":
        lineplot(xd, _ax=ax, data=data)
    elif sd == "Scatter Plot":
        scatterplot(xd, _ax=ax, data=data)
    elif sd == "Count Plot":
        countplot(xd, _ax=ax, data=data)
    elif sd == "Strip Plot":
        stripplot(xd, _ax=ax, data=data)
    elif sd == "Density Plot":
        densityplot(xd, _ax=ax, data=data, shade=shade)
    return fig

if __name__ == "__main__":
    
    st.sidebar.header("Visualization")
    st.markdown("# Boston House Prices")

    samples = st.sidebar.slider("samples", max_value=len(df), value=20, step=10)
    df_samples = get_sample_dataframe(samples)


    sd = st.sidebar.selectbox(
        "Select type of plot", 
        [
            "Line Plot",
            "Count Plot",
            "Scatter Plot",
            "Strip Plot",
            "Density Plot",
            "Joint Plot",
            "Pair Plot"
            
        ]
    )       
        
    if sd in ['Pair Plot', "Joint Plot"]:
        kindd = st.sidebar.selectbox(
            "Select the kind of the graph",
            [
                "kde",
                "scatter",
                "hist",
                "reg",
                "hex",
                "resid"
            ]
        )

        if sd == "Joint Plot":
            xd = st.sidebar.selectbox(
                "Select X-axis", df.columns
            )
            if kindd == "kde":
                shade = st.sidebar.checkbox("shade", False)
                fig = get_plot(sd=sd, xd=xd, kind=kindd, data=df_samples, shade=shade)
            else:
                fig = get_plot(sd=sd, xd=xd, kind=kindd, data=df_samples)
        else:
            if kindd == "kde":
                shade = st.sidebar.checkbox("shade", True)
                fig = pairplot(data=df_samples, kind=kindd, shade=shade).figure
            else:
                fig = pairplot(data=df_samples, kind=kindd).figure
            # fig = sns.pairplot(data=df_samples, kind=kindd, diag_kws={"shade":True}).figure
    else:
        xd = st.sidebar.selectbox(
                "Select X-axis", df.columns[:-1]
            )
        if sd == "Density Plot":
            shade = st.sidebar.checkbox("shade", False)
            fig = get_plot(sd=sd, xd=xd, data=df_samples, shade=shade)
        else:
            fig = get_plot(sd=sd, xd=xd, data=df_samples)


    if sd == "Pair Plot":
        fig_to_html = mpld3.fig_to_html(fig)
        components.html(fig_to_html, height=2000, width=1350, scrolling=True)
    else:
        col1, col2 = st.columns(2)

        with col1:
            fig_to_html = mpld3.fig_to_html(fig)
            components.html(fig_to_html, height=2000, width=800, scrolling=True)

        with col2:
            with st.container():
                st.write("## Observation")
            with st.expander(observations[xd]['title'].upper()):
                st.write(str(observations[xd]['obs'].format(sd)))
