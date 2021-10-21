import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np

#Crear entorno virtual :
#   conda create -n streamlit -y

#Inicializar el entorno virtual:
#   conda activate streamlit

#Ejecutamos con:
#   streamlit run mi_programa.py [-- otros posibles argumentos]
#

st.title("Algoritmos de Inteligencia Artificial")

menu = st.sidebar.selectbox(
    label = "Algoritmos",
    options = ["Cluster", "Cluster k-means"],
    index = 0,
)
if menu == "Cluster":
    st.title("Cluster")
    
elif menu == "Cluster k-means":
    st.title("Cluster k-means") 

if st.sidebar.button("Escribir"):
    st.write("**Has picado el botón**")
else:
    st.write("")

if st.sidebar.button("Limpiar"):
    st.write("")
else:
    st.write("")

