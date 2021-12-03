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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator

def impresion():
    tipoGraf = 0
    col1, col2, col3 = st.columns(3)
    with col2:
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

    #----------------SELECCION DE VARIABLES----------------------------
        with st.form("my_form"):
            st.header('Seleccion de variables')
            variables = st.multiselect('Selecciona las variabels para el algoritmo',
                                    Encabezado, Encabezado[3])
            Matriz = np.array(Datos[variables])
            submitted = st.form_submit_button("Matriz con las variables")
            if submitted:
                st.write(pd.DataFrame(Matriz))
    #----------------ESTANDARIZACION DE DATOS---------------------------
        estandarizar = StandardScaler()
        MEstandarizada = estandarizar.fit_transform(Matriz)
    #----------------K CLUSTERS-----------------------------------------
        SSE = []
        for i in range(2, 12):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(MEstandarizada)
            SSE.append(km.inertia_)
        kl = KneeLocator(range(2,12), SSE, curve = "convex", direction = "decreasing")
        
        with st.expander("Informacion de clusters recomendados"):
            st.subheader('El algoritmo te recomeinda que elijas '+str(kl.elbow)+
                        ' clusters para el algoritmo')
            kl.plot_knee()
            st.pyplot()

        nclusters = st.number_input('Inserte el numero de clusters que desea tener', step=1, min_value = 2)
        
        if nclusters != 0:
            clustern = st.slider(
            'Selecciona el numero del cluster que quieras ver',
            0, nclusters-1)
#--------------ALGORITMO-------------------------------------------
        if st.button('Aplicar algoritmo'):
            MParticional  = KMeans(n_clusters=int(nclusters), random_state=0).fit(MEstandarizada)
            MParticional.predict(MEstandarizada)
            Datos['Numero_Cluster'] = MParticional.labels_
#-------------RESULTADOS--------------------------------------------
            st.subheader('Tus datos con el numero de clusters asignado:')
            st.write(Datos)
            st.subheader('Tabla del cluster '+ str(clustern))
            st.write(Datos[Datos.Numero_Cluster == clustern])
            st.subheader('Número de elementos por cluster')
            st.write(Datos.groupby(['Numero_Cluster'])['Numero_Cluster'].count())
            st.subheader('Centroides')
            CentroidesP = Datos.groupby(['Numero_Cluster'])[variables].mean()
            st.write(CentroidesP)
    #------------------------BOTONES SIDEBAR---------------------------
        if st.sidebar.button("Ver datos"):
            st.subheader('Primeros 10 datos\n ')
            st.write(Datos.head(10))
