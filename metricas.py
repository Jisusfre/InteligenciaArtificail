import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from PIL import Image as im

def impresion():
    st.title("Metricas de distancia")
    imagen = im.open('Imagenes\MetricasDistancia.png')
    st.image(imagen, caption = 'Caminos del Punto A al punto B')

#--------------------Lectura de datos-------------------
    st.subheader('Elige el archivo con los datos a trabajar para iniciar\n ')
    Datos_subidos = st.file_uploader(" ", type = 'csv')

    if Datos_subidos is not None:
        #-------------------DATOS---------------------------------
        Datos = pd.read_csv(Datos_subidos)
        #-------------------MATRIZ EUCLIDEANA---------------------
        DstEuclidiana = cdist(Datos, Datos, metric='euclidean')
        MEuclidiana = pd.DataFrame(DstEuclidiana)
        #-------------------MATRIZ CHEBYSHEV----------------------
        DstChebyshev = cdist(Datos, Datos, metric='chebyshev')
        MChebyshev = pd.DataFrame(DstChebyshev)
        #-------------------MATRIZ CITYBLOCK----------------------
        DstCityblock = cdist(Datos, Datos, metric='cityblock')
        MCityblock = pd.DataFrame(DstCityblock)
        #-------------------MATRIZ MINKOWSKI----------------------
        DstMinkowski = cdist(Datos, Datos, metric='minkowski')
        MMinkowski = pd.DataFrame(DstMinkowski)

        st.header('Matrices de distancias:')
        if st.button('Euclideana'):
            st.write(MEuclidiana)

        if st.button('Chebyshev'):
            st.write(MChebyshev)

        if st.button('Cityblock'):
            st.write(MCityblock)
        
        if st.button('Minkowski'):
            st.write(MMinkowski)

        st.header('Distancia entre dos objetos')
        st.subheader('Elige el índice de los 2 objetos a comparar la distancia:')

        objeto1 = st.number_input('Inserte el objeto 1', step=1)
        st.subheader('El objeto 1 es el número '+ str(objeto1))

        objeto2 = st.number_input('Inserte el objeto 2', step=1)
        st.subheader('El objeto 2 es el número '+ str(objeto2))

        if objeto1 != 0 and objeto2 != 0:
            if st.button('Distancia Euclideana'):
                Objeto1 = Datos.iloc[objeto1]
                Objeto2 = Datos.iloc[objeto2]
                dstEuclidiana = distance.euclidean(Objeto1,Objeto2)
                st.subheader('La distancia entre los objetos es de ' + str(dstEuclidiana)[:-9])

            if st.button('Distancia Chebyshev'):
                Objeto1 = Datos.iloc[objeto1]
                Objeto2 = Datos.iloc[objeto2]
                dstChebyshev = distance.chebyshev(Objeto1,Objeto2)
                st.subheader('La distancia entre los objetos es de ' + str(dstChebyshev))

            if st.button('Distancia Cityblock'):
                Objeto1 = Datos.iloc[objeto1]
                Objeto2 = Datos.iloc[objeto2]
                dstCityblock = distance.cityblock(Objeto1,Objeto2)
                st.subheader('La distancia entre los objetos es de ' + str(dstCityblock))
        
            if st.button('Distancia Minkowski'):
                Objeto1 = Datos.iloc[objeto1]
                Objeto2 = Datos.iloc[objeto2]
                dstMinkowski = distance.minkowski(Objeto1,Objeto2,p=1.5)
                st.subheader('La distancia entre los objetos es de ' + str(dstMinkowski)[:-4])
        
#------------------------BOTONES SIDEBAR---------------------------
        if st.sidebar.button("Ver datos"):
            st.subheader('Primeros 10 datos\n ')
            st.write(Datos.head(10))