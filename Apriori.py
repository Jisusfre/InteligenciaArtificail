import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np

from apyori import apriori

def impresion():
    st.title("Algoritmo Apriori")
    DatosMovies = pd.read_csv("Datos/movies.csv")


    if st.sidebar.button("Ver datos"):
        st.subheader('Primeros 10 datos\n ')
        st.write(DatosMovies.head(10))

        if st.sidebar.button("Limpiar"):
            st.write("")
        else:
            st.write("")
    else:
        st.write("")

    
    