import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

#reads data
df = pd.read_csv('Data\editResultStorage.csv')

#make two lists of unique ears and plots
availablePlots = df['ID'].unique()
availableEars = df['Out'].unique()

st.header('Wiew test results here')

#Choose id to show
idChosen = st.selectbox('Select data to show',availablePlots)
#Choose ear to show
earChosen = st.selectbox('Select data to show',availableEars)

dff = df[df['ID'] == idChosen]  
dfff = dff[dff['Out'] == earChosen]
dfff["Frekvens"] = dfff["Frekvens"].astype(str)
fig, ax = plt.subplots()
ax.scatter(dff['Frekvens'], dff["AnswerTime"],c=dff["Response"])
ax.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",)
ax.update_traces(marker=dict(size=18,
                  line=dict(width=2,
                    color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
st.pyplot(fig)