import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np
#Importar otras páginas
import Apriori
import ClusterJerarquico
import Inicio
import metricas
import ClusterPar
import Clasificacion
import ArbolPronostico
import ArbolDecision
import Modulo
#Crear entorno virtual :
#   conda create -n streamlit -y

#Inicializar el entorno virtual:
#   conda activate streamlit

#Desactivar el entorno virtual
#conda deactivate

#Ejecutamos con:
#   streamlit run mi_programa.py [-- otros posibles argumentos]

#Variables
PAGES = {"Apriori":Apriori, "Cluster Jerarquico":ClusterJerarquico, "Inicio":Inicio,
         "Metricas":metricas, "Cluster Particional": ClusterPar, "Clasificacion":Clasificacion,
         "Arbol Pronostico":ArbolPronostico, "Arbol de Decision":ArbolDecision}

#--------------------------------------------------------------------------
st.set_page_config(
    layout = "wide",
    page_icon = '🧠'
)
st.set_option('deprecation.showPyplotGlobalUse', False)

menu = st.sidebar.selectbox(
    label = "Algoritmos",
    options = ["Inicio", "Apriori", "Metricas","Cluster Jerarquico", 
                "Cluster Particional", "Clasificacion", "Arbol Pronostico",
                "Arbol de Decision"],
    index = 0,
)
if menu == "Apriori":
    page = PAGES[menu]
    page.impresion()
    
elif menu == "Cluster Jerarquico":
    page = PAGES[menu]
    page.impresion()

elif menu == "Inicio":
    page = PAGES[menu]
    page.entrada()

elif menu == "Metricas":
    page = PAGES[menu]
    page.impresion()

elif menu == "Cluster Particional":
    page = PAGES[menu]
    page.impresion()

elif menu == "Clasificacion":
    page = PAGES[menu]
    page.impresion()

elif menu == "Arbol Pronostico":
    page = PAGES[menu]
    page.impresion()

elif menu == "Arbol de Decision":
    page = PAGES[menu]
    page.impresion()

