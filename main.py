import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np
#Importar otras páginas
import Apriori
import Cluster
import Inicio
#Crear entorno virtual :
#   conda create -n streamlit -y

#Inicializar el entorno virtual:
#   conda activate streamlit

#Desactivar el entorno virtual
#conda deactivate

#Ejecutamos con:
#   streamlit run mi_programa.py [-- otros posibles argumentos]

#Variables
PAGES = {"Apriori":Apriori, "Cluster":Cluster, "Inicio":Inicio}

#--------------------------------------------------------------------------

menu = st.sidebar.selectbox(
    label = "Algoritmos",
    options = ["Inicio", "Apriori", "Cluster"],
    index = 0,
)
if menu == "Apriori":
    page = PAGES[menu]
    page.impresion()
    
elif menu == "Cluster":
    page = PAGES[menu]
    page.impresion()

elif menu == "Inicio":
    page = PAGES[menu]
    page.entrada()

if st.sidebar.button("Escribir"):
    st.write("**Has picado el botón**")
else:
    st.write("")

if st.sidebar.button("Limpiar"):
    st.write("")
else:
    st.write("")

