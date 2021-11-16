import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np
from PIL import Image as im

from apyori import apriori


def impresion():
    st.title("Algoritmo Apriori")
    imagen = im.open('Imagenes\Arpiori.jpg')
    st.image(imagen, caption = 'Items')
    #DatosMovies = pd.read_csv("Datos/movies.csv")
    st.subheader('Elige el archivo con los datos a trabajar\n ')
    Datos_subidos = st.file_uploader(" ", type = 'csv')

    if Datos_subidos is not None:
        Datos = pd.read_csv(Datos_subidos)
        if st.sidebar.button("Iniciar"):
            SeleccionDatos(Datos)
        if st.sidebar.button("Ver datos"):
            st.subheader('Primeros 10 datos\n ')
            st.write(Datos.head(10))
        
    


def SeleccionDatos(Datos):
    Transaccion = Datos.values.reshape(-1).tolist()
    ListaM = pd.DataFrame(Transaccion)
    ListaM['Frecuencia'] = 0
    ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'],ascending=True) #Conteo
    ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
    ListaM = ListaM.rename(columns={0 : 'Item'})
    st.subheader('Creando lista con todas las transacciones')
    st.write(ListaM)

    
        
    
    