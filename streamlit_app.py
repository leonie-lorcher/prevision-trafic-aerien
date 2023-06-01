pip install plotly
pip install prophet

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px
from prophet import Prophet

traffic_df = pd.read_parquet('https://github.com/leonie-lorcher/prevision-trafic-aerien/blob/main/data/traffic_10lines.parquet')
traffic_df