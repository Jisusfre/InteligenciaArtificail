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
    st.title("Cluster Jerarquico")
    imagen = im.open('Imagenes\ClustersJerar.jpg')
    st.image(imagen, caption = 'Caminos del Punto A al punto B')

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
            grafica = sns.pairplot(Datos, hue = variable)
            st.pyplot(grafica)
        elif tipoGraf == 2:
            fig, ax = plt.subplots(figsize=(18,15), dpi=300)
            MatrizInf = np.triu(CorrDatos)
            sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
            st.pyplot(fig)
        else:
            pass
        
    #----------------SELECCION DE VARIABLES----------------------------
        st.header('Seleccion de variables')
        variables = st.multiselect('Selecciona las variabels para el algoritmo',
                                  Encabezado, Encabezado[0])
        Matriz = np.array(Datos[variables])
        if st.button('Matriz con las variables'):
            st.write(pd.DataFrame(Matriz))
    #------------------------APLICACION DEL ALGORITMO------------------
        nclusters = st.number_input('Inserte el numero de clusters que desea tener', step=1)
        
        if nclusters != 0:
            clustern = st.slider(
            'Selecciona el numero del cluster que quieras ver',
            0, nclusters-1)
            if st.button('Aplicar algoritmo'):
                estandarizar = StandardScaler()                                
                MEstandarizada = estandarizar.fit_transform(Matriz)
                st.header('Árbol de clusters')
                fig, ax = plt.subplots(figsize=(25,15), dpi=300)
                ax.set_ylabel('Distancia')
                ax.set_xlabel('Hipoteca')
                shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
                st.pyplot(fig)   
        #--------------------------------NUMERO DE CLUSTERS----------------
                MJerarquico = AgglomerativeClustering(n_clusters=nclusters, linkage='complete', affinity='euclidean')
                MJerarquico.fit_predict(MEstandarizada)
                st.subheader('Tus datos con el numero de clusters asignado:')
                Datos['ClusterH'] = MJerarquico.labels_
                st.write(Datos)
                st.subheader('Tabla del cluster '+ str(clustern))
                st.write(Datos[Datos.ClusterH == clustern])
                st.subheader('Número de elementos por cluster')
                st.write(Datos.groupby(['ClusterH'])['ClusterH'].count())

    #------------------------BOTONES SIDEBAR---------------------------
        if st.sidebar.button("Ver datos"):
            st.subheader('Primeros 10 datos\n ')
            st.write(Datos.head(10))
        