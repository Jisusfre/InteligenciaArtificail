import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PIL import Image as im
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

def impresion():
    tipoGraf = 0
    st.title("Cluster Particional")
    imagen = im.open('Imagenes\ClusterPart.jpg')
    st.image(imagen, caption = 'Cluster')

    #--------------------Lectura de datos--------------------------------------
    st.subheader('Elige el archivo con los datos a trabajar para iniciar\n ')
    Datos_subidos = st.file_uploader(" ", type = 'csv')
    
    if Datos_subidos is not None:
    #-------------------DATOS------------------------------------------
        Datos = pd.read_csv(Datos_subidos)
    #-------------------MATRIZ DE CORRELACION--------------------------
        CorrDatos = Datos.corr(method = 'pearson')
    #-------------------GRAFICOS---------------------------------------
        st.header('Graficas')
        Encabezado = list(Datos.columns)
        variable = st.selectbox(
        'Elige la variable para el grafico de dispersión',
        Encabezado)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Grafica de dispresión"):
                tipoGraf = 1 
        
        with col2:
            if st.button("Mapa de calor"):
                tipoGraf = 2 

        if tipoGraf == 1:
            with st.spinner('Cargando la grafica de dispersión...'):
                grafica = sns.pairplot(Datos, hue = variable)
                st.pyplot(grafica)
        elif tipoGraf == 2:
            with st.spinner('Cargando el mapa de calor...'):
                fig, ax = plt.subplots(figsize=(14,10), dpi=200)
                MatrizInf = np.triu(CorrDatos)
                sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
                st.pyplot(fig)
        else:
            pass

    #------------------------BOTONES SIDEBAR---------------------------
        if st.sidebar.button("Ver datos"):
            st.subheader('Primeros 10 datos\n ')
            st.write(Datos.head(10))
    #----------------SELECCION DE VARIABLES----------------------------
        st.header('Seleccion de variables')
        variables = st.multiselect('Selecciona las variabels para el algoritmo',
                                  Encabezado, Encabezado[0])
        Matriz = np.array(Datos[variables])
        if st.button('Matriz con las variables'):
            st.write(pd.DataFrame(Matriz))