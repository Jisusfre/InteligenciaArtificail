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
    error = 0
    matriz = 0
    col1, col2, col3 = st.columns(3)
    with col2:
        st.title("Metricas de distancia")
        imagen = im.open('Imagenes\MetricasDistancia.png')
        st.image(imagen, caption = 'Distancias del Punto A al punto B')
    
    #--------------------Lectura de datos-------------------
    st.subheader('Elige el archivo con los datos a trabajar para iniciar\n ')
    Datos_subidos = st.file_uploader(" ", type = 'csv')

    if Datos_subidos is not None:
        try:
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
        except ValueError:
            st.header('Archivo no soportado')
            error = 1

#------------------------------------------MATRICES---------------------------------------
        if error == 0: 
            st.header('Matrices de distancias:')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button('Euclideana'):
                    matriz = 1

            with col2:
                if st.button('Chebyshev'):
                    matriz = 2
            
            with col3:
                if st.button('Cityblock'):
                    matriz = 3

            with col4:
                if st.button('Minkowski'):
                    matriz = 4   
            
            if matriz == 1:
                st.write(MEuclidiana)
            elif matriz == 2:
                st.write(MChebyshev)
            elif matriz == 3:
                st.write(MCityblock)
            elif matriz == 4:
                st.write(MMinkowski)  
            else:
                pass

#--------------------------------------Distancias--------------------------------------------------
            st.header('Distancia entre dos objetos')
            st.subheader('Elige el índice de los 2 objetos a comparar la distancia:')
            col1, col2 = st.columns(2)
            with col1:
                objeto1 = st.number_input('Inserte el numero del objeto 1', step=1)
                st.subheader('El objeto 1 es el número '+ str(objeto1))
            
            with col2:
                objeto2 = st.number_input('Inserte el numero del objeto 2', step=1)
                st.subheader('El objeto 2 es el número '+ str(objeto2))

            if objeto1 != 0 and objeto2 != 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    with st.form('DistanciasEuclideana'):     
                        submitted = st.form_submit_button('Distancia Euclideana')
                        if submitted:
                            Objeto1 = Datos.iloc[objeto1]
                            Objeto2 = Datos.iloc[objeto2]
                            dstEuclidiana = distance.euclidean(Objeto1,Objeto2)
                            st.info('La distancia entre los objetos es de ' + str(dstEuclidiana)[:-9])
                
                with col2:
                    with st.form('DistanciasChebyshev'):
                        submitted = st.form_submit_button('Distancia Chebyshev')
                        if submitted:
                            Objeto1 = Datos.iloc[objeto1]
                            Objeto2 = Datos.iloc[objeto2]
                            dstChebyshev = distance.chebyshev(Objeto1,Objeto2)
                            st.info('La distancia entre los objetos es de ' + str(dstChebyshev))
                
                with col3:
                    with st.form('DistanciasCityblock'):                    
                        submitted = st.form_submit_button('Distancia Cityblock')
                        if submitted:
                            Objeto1 = Datos.iloc[objeto1]
                            Objeto2 = Datos.iloc[objeto2]
                            dstCityblock = distance.cityblock(Objeto1,Objeto2)
                            st.info('La distancia entre los objetos es de ' + str(dstCityblock))

                with col4:
                    with st.form('DistanciasMinkowski'):                    
                        submitted = st.form_submit_button('Distancia Minkowski')
                        if submitted:
                            Objeto1 = Datos.iloc[objeto1]
                            Objeto2 = Datos.iloc[objeto2]
                            dstMinkowski = distance.minkowski(Objeto1,Objeto2,p=1.5)
                            st.info('La distancia entre los objetos es de ' + str(dstMinkowski)[:-4])
    
#------------------------BOTONES SIDEBAR---------------------------
        if st.sidebar.button("Ver datos"):
            st.subheader('Primeros 10 datos\n ')
            st.write(Datos.head(10))
