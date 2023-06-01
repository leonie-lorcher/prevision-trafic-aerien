import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px
import prophet
from prophet import Prophet
from datetime import datetime


st.title('Outil de prévision du trafic aérien')

st.image('https://www.sunshinecoastairport.com.au/wp-content/uploads/2017/08/flights-radar-header.jpg')

st.caption("Cet outil a été développé dans le but de produire une prédiction de la fréquentation de lignes aériennes, en nombre total de passagers. Vous pourrez choisir la ligne qui vous intéresse, étudier les évolutions de fréquentation qu'elle a connues, et visualiser la prédiction de trafic, sur le nombre de jours choisis. Bonne visite !")

parquet_file = 'https://github.com/leonie-lorcher/prevision-trafic-aerien/raw/main/data/traffic_10lines.parquet'
traffic_df = pd.read_parquet(parquet_file, engine='auto')

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.subheader("1. Choix de la ligne aérienne")

dict1, dict2, dict3 = st.columns(3)
with dict1:
	st.text('AMS = Amsterdam')
	st.text('BCN = Barcelone')
	st.text('FUE = Fuertaventura')
	st.text('GRU = Sao Paulo')
	st.text('JFK = New York')
	st.text('LHR = Londres Heathrow')
with dict2:
	st.text('LIS = Lisbonne')
	st.text('LGW = Londres Gatwick')
	st.text('LYS = Lyon')
	st.text('NGB = Ningbo')
	st.text('OPO = Porto')
	st.text('ORY = Paris')
with dict3:
	st.text('PIS = Poitiers')
	st.text('PNH = Phnom Penh')
	st.text('POP = Puerto Plata')
	st.text('SCL = Santiago')
	st.text('SSA = Bahia')

st.markdown(" **Choisir la ligne aérienne souhaitée :** ")

ligne_aer = st.selectbox(
	"Ligne aérienne :",
	('LIS-OPO', 'LIS-ORY', 'LGW-BCN', 'LGW-AMS', 'SSA-GRU', 'POP-JFK', 'SCL-LHR', 'NTE-FUE', 'LYS-PIS', 'PNH-LGB'),
   	label_visibility = st.session_state.visibility,
   	disabled = st.session_state.disabled
)

if ligne_aer == 'LIS-OPO':
	aer_dep = 'LIS'
	aer_arriv = 'OPO'
if ligne_aer == 'LIS-ORY':
	aer_dep = 'LIS'
	aer_arriv = 'ORY'
if ligne_aer == 'LGW-BCN':
	aer_dep = 'LGW'
	aer_arriv = 'BCN'
if ligne_aer == 'LGW-AMS':
	aer_dep = 'LGW'
	aer_arriv = 'AMS'
if ligne_aer == 'SSA-GRU':
	aer_dep = 'SSA'
	aer_arriv = 'GRU'
if ligne_aer == 'POP-JFK':
	aer_dep = 'POP'
	aer_arriv = 'JFK'
if ligne_aer == 'SCL-LHR':
	aer_dep = 'SCL'
	aer_arriv = 'LHR'
if ligne_aer == 'NTE-FUE':
	aer_dep = 'NTE'
	aer_arriv = 'FUE'
if ligne_aer == 'LYS-PIS':
	aer_dep = 'LYS'
	aer_arriv = 'PIS'
if ligne_aer == 'PNH-LGB':
	aer_dep = 'PNH'
	aer_arriv = 'LGB'


# Création d'une fonction qui génère un dataframe contenant le nombre de passagers, par date, sur une ligne aérienne donnée.
def generate_route_df(traffic_df: pd.DataFrame, homeAirport: str, pairedAirport: str) -> pd.DataFrame:
  _df = (traffic_df
         .query('home_airport == "{home}" and paired_airport == "{paired}"'.format(home=homeAirport, paired=pairedAirport))
         .groupby(['home_airport', 'paired_airport', 'date'])
         .agg(pax_total=('pax', 'sum'))
         .reset_index()
         )
  return _df

# On génère un dataframe pour la ligne choisie
ligne_aer_df = generate_route_df(traffic_df, aer_dep, aer_arriv)
ligne_aer_df = ligne_aer_df.rename(columns={'date': 'ds', 'pax_total': 'y'})

st.subheader("2. Évolution du trafic")

st.caption("Le curseur ci-dessous vous permet de sélectionner les dates qui vous intéressent plus facilement. Cela impactera également le graph de la partie 3.")

# Récupérer le min et le max des dates dans le dataset
min_date = ligne_aer_df['ds'].min().strftime('%Y-%m-%d')
min_date = datetime.strptime(min_date, '%Y-%m-%d').date()

max_date = ligne_aer_df['ds'].max().strftime('%Y-%m-%d')
max_date = datetime.strptime(max_date, '%Y-%m-%d').date()

# Créer le date slider
selected_dates = st.slider(
    "Sélecteur de dates",
    min_value = min_date,
    max_value = max_date,
    value = [min_date, max_date]
)

min_date = pd.to_datetime(selected_dates[0])
max_date = pd.to_datetime(selected_dates[1])

filtered_df = ligne_aer_df[(ligne_aer_df['ds'] >= min_date) & (ligne_aer_df['ds'] <= max_date) ]

st.markdown(" **Série temporelle du nombre de passagers ayant voyagé sur la ligne aérienne sélectionnée :** ")

# Graph du trafic sur la ligne sélectionnée
fig = px.line(filtered_df, x = 'ds', y = 'y', title = 'Trafic sur la ligne aérienne sélectionnée')
fig.update_traces(line_color = 'skyblue')
fig.update_layout(xaxis_title = 'Date', yaxis_title = 'Nombre total de passagers')

st.plotly_chart(fig, use_container_width = True)

st.subheader("3. Prédiction du trafic")

# Entraînement du modèle sur le dataframe
baseline_model = Prophet()
baseline_model.fit(ligne_aer_df)

nb_periods = st.number_input('Combien de jours voulez-vous prédire ?', min_value=15, max_value=80, step=1)

# Génère un dataframe de dates, avec 15 jours en plus
future_df = baseline_model.make_future_dataframe(periods = nb_periods)

# Produire une prédiction du nombre total de passagers (pax total) sur les 15 prochains jours
forecast_df = baseline_model.predict(future_df)

# Dans le dataframe de prédiction, garder uniquement les données prédites pour le graph.
forecast_ddf = forecast_df[['ds', 'yhat']].tail(nb_periods)

filtered_forecast_df = forecast_ddf[(forecast_ddf['ds'] >= min_date) & (forecast_ddf['ds'] <= max_date) ]

# Graph
fig = px.line(filtered_df, x = 'ds', y = 'y', title = 'Historique et prédiction du trafic sur la ligne aérienne')
fig.update_traces(line_color = 'skyblue', name = 'Historique', showlegend = True)
fig.add_scatter(x = filtered_forecast_df['ds'], y = filtered_forecast_df['yhat'], mode = 'lines', name = 'Prédiction', line = dict(color='salmon'))
fig.update_layout(xaxis_title = 'Date', yaxis_title = 'Nombre total de passagers')

# Création du bouton

if st.button('Cliquer ici pour générer la prédiction'):
    st.plotly_chart(fig, use_container_width = True)
else:
    st.write(' ')

st.divider()

st.caption("Léonie LORCHER (Master 2 Econométrie, Big Data, Statistiques - Aix-Marseille School of Economics)")








