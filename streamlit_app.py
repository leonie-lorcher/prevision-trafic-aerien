#pip install plotly
#pip install prophet

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px
import prophet
from prophet import Prophet

parquet_file = 'https://github.com/leonie-lorcher/prevision-trafic-aerien/raw/main/data/traffic_10lines.parquet'
traffic_df = pd.read_parquet(parquet_file, engine='auto')
traffic_df