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
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.title("Cluster Jerarquico")
        imagen = im.open('Imagenes\ClustersJerar.jpg')
        st.image(imagen, caption = 'Cluster', width = 400)

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
            with st.spinner('Cargando la grafica de dispersión... esto puede tardar unos segundos'):
                grafica = sns.pairplot(Datos, hue = variable)
                st.pyplot(grafica)
            
        elif tipoGraf == 2:
            with st.spinner('Cargando el mapa de calor...'):
                fig, ax = plt.subplots(figsize=(18,15), dpi=300)
                MatrizInf = np.triu(CorrDatos)
                sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
                st.pyplot(fig)
        else:
            pass
        
    #----------------SELECCION DE VARIABLES----------------------------
        with st.form("my_form"):
            st.header('Seleccion de variables')
            variables = st.multiselect('Selecciona las variables para el algoritmo',
                                    Encabezado, Encabezado[0])
            Matriz = np.array(Datos[variables])
            submitted = st.form_submit_button("Matriz con las variables")
            if submitted:
                st.write(pd.DataFrame(Matriz))
        col1, col2, col3 = st.columns(3)
        with col2:
            nclusters = st.number_input('Inserte el numero de clusters que desea tener', step=1)
        
        if nclusters != 0:
            clustern = st.slider(
            'Selecciona el numero del cluster que quieras ver',
            0, nclusters-1)
    #------------------------APLICACION DEL ALGORITMO------------------
            if st.button('Aplicar algoritmo'):
                estandarizar = StandardScaler()                                
                MEstandarizada = estandarizar.fit_transform(Matriz)
                with st.expander("Arbol"):
                    with st.spinner('Graficando el arbol... esto puede tardar unos segundos'):
                        fig, ax = plt.subplots(figsize=(25,15), dpi=300)
                        ax.set_ylabel('Distancia')
                        ax.set_xlabel(Datos_subidos.name)
                        shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
                        st.pyplot(fig)   
        #--------------------------------NUMERO DE CLUSTERS----------------
                MJerarquico = AgglomerativeClustering(n_clusters=nclusters, linkage='complete', affinity='euclidean')
                MJerarquico.fit_predict(MEstandarizada)
                st.subheader('Tus datos con el numero de clusters asignado:')
                Datos['Numero_Cluster'] = MJerarquico.labels_
                st.write(Datos)
                st.subheader('Tabla del cluster '+ str(clustern))
                st.write(Datos[Datos.Numero_Cluster == clustern])
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.subheader('Número de elementos por cluster')
                    st.write(Datos.groupby(['Numero_Cluster'])['Numero_Cluster'].count())

    #------------------------BOTONES SIDEBAR---------------------------
        if st.sidebar.button("Ver datos"):
            st.subheader('Primeros 10 datos\n ')
            st.write(Datos.head(10))
        