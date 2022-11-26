import streamlit as st
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor 
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from lightgbm import LGBMRegressor 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Prediction",
    page_icon="P",
    layout="wide"
)

from main import load_dataset
df = load_dataset()

@st.cache
def get_sample_dataframe(samples=10):
    return df.sample(samples)


def report_metric(pred, test, model_name):
    # Creates report with mae, rmse, mse and r2 metric and returns Dataframe
    mae = mean_absolute_error(pred, test)
    mse = mean_squared_error(pred, test)
    rmse = np.sqrt(mse)
    r2 = r2_score(test, pred)

    metric_data = {"Metric": ["MAE", "MSE", "RMSE", "R2"], model_name: [mae, mse, rmse, r2]}
    metric_df = pd.DataFrame(metric_data)

    #if "metric" not in st.session_state:
    
    return metric_df 

@st.experimental_memo
def plot_preds(real_data, test_data, target, pred):
    # Plots prediction vs Real

    figs = []
    for i in range(2):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(real_data[real_data.columns[i]], target, label = f"Real {real_data.columns[i]}")
        ax.plot(test_data[test_data.columns[i]], pred, label=f"Pred {test_data.columns[i]}")
        ax.legend()
        figs.append(fig)
    return figs

@st.experimental_memo
def model_prep(data, columns, target, model_name, *args, **kwargs):
    X, y = data[columns], data[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if model_name == "LGBM Regressor":
        norm = MinMaxScaler().fit(x_train)
        x_train = pd.DataFrame(norm.transform(x_train))
        x_test = pd.DataFrame(norm.transform(x_test))
        x_train.columns = x_test.columns = columns

    model = eval(f"{''.join(model_name.split())}(*{args}, **{kwargs})")
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metric = report_metric(pred=pred, test=y_test, model_name=model_name)
    return metric, X, x_test, y, pred 

def get_correlation_values(data=round(df, 2)):
    s = {col: round(data["medv"].corr(data[col]), 2) for col in df.columns if col != "medv"}
    result = min(s, key=s.get), max(s, key=s.get)
    return {res: s[res] for res in result}


st.sidebar.header("Prediction")
st.markdown("# Boston House Prices")

if __name__ == "__main__":

    #samples = st.sidebar.slider("samples", min_value=20, max_value=len(df)//2, step=20)
    # df_samples = get_sample_dataframe(samples)

    if "corr" not in st.session_state:
        st.session_state['corr'] = get_correlation_values() 
    corr = st.session_state['corr']

    model_name = st.sidebar.selectbox(
        "Select the appropriate Model",
        [
            "Linear Regression",
            "XGB Regressor",
            "LGBM Regressor",
        ]
    )
    select_all = st.sidebar.checkbox("Select All", True)
    columns = st.sidebar.multiselect(
        "Select independent columns",
        df.columns[:-1], default=corr
    )
    # args and kwargs
    kwargs = {}
    if model_name == "XGB Regressor":
        n_estimators = st.sidebar.number_input("Estimator Number", min_value=50, max_value=1000, value=100, step=50)
        learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.05, max_value=1.0, step=0.05, value=0.05)
        kwargs['n_estimators'] = n_estimators
        kwargs['learning_rate'] = learning_rate
    elif model_name == "LGBM Regressor":
        n_estimators = st.sidebar.number_input("Estimator Number", min_value=50, max_value=1000, value=100, step=50)
        learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.05, max_value=1.0, step=0.05, value=0.05)
        max_depth = st.sidebar.number_input("Maximum Depth", min_value=2, max_value=20, value=2, step=2)
        kwargs['n_estimators'] = n_estimators
        kwargs['learning_rate'] = learning_rate
        kwargs['max_depth'] = max_depth
        kwargs['min_child_samples'] = 25
        kwargs['num_leaves'] = 31

    st.markdown(f"## {model_name}")
    
    metric_data, real_data, test_data, target_data, pred_data = model_prep(df, columns, ['medv'], model_name, **kwargs)
    st.dataframe(metric_data)
    if 'metric' not in st.session_state:
        st.session_state['metric'] = metric_data.copy()
    st.session_state['metric'][model_name] = metric_data[model_name].copy()
    figs = plot_preds(real_data=real_data[corr], test_data=test_data[corr], target=target_data, pred=pred_data)

    # columns
    col_1, col_2 = st.columns(2, gap="medium")

    for fig, col in zip(figs, [col_1, col_2]):
        with col:
            fig_html = mpld3.fig_to_html(fig)
            components.html(fig_html, scrolling=False, height=1050)