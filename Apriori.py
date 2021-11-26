import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np
from PIL import Image as im

from apyori import apriori
#------------Variables-------------------------
Reglas = []
Confianza = []
Soporte = []
Lift = []

#------------Codigo-------------------------------
def impresion():
    st.title("Algoritmo Apriori")
    imagen = im.open('Imagenes\Arpiori.jpg')
    st.image(imagen, caption = 'Items')
    #DatosMovies = pd.read_csv("Datos/movies.csv")
    st.subheader('Elige el archivo con los datos a trabajar para iniciar\n ')
    Datos_subidos = st.file_uploader(" ", type = 'csv')

    if Datos_subidos is not None:
        #Datos
        Datos = pd.read_csv(Datos_subidos, header = None)
        #Transaccion y ListaM
        Transaccion = Datos.values.reshape(-1).tolist()
        ListaM = pd.DataFrame(Transaccion)
        #ListaM con Frecuencia y porcentaje
        ListaM['Frecuencia'] = 0
        ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'],ascending=True) #Conteo
        ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
        ListaM = ListaM.rename(columns={0 : 'Item'})
        #lista
        lista = Datos.stack().groupby(level=0).apply(list).tolist()
        #Obtenemos la confianza, soporte y elevacion
        soporte = st.number_input('Inserte el soporte')
        st.subheader('El soporte que elegiste es de '+ str(soporte))

        confianza = st.number_input('Inserte la confianza')
        st.subheader('La confinza que elegiste es de '+ str(confianza))

        elevacion = st.number_input('Inserte la elevacion')
        st.subheader('La elevacion que elegiste es de '+ str(elevacion))
        #Aplicamos el algoritmo
        if st.button('Aplicar algoritmo'):
            Reglas = []
            Confianza = []
            Soporte = []
            Lift = []
            ReglasC1 = apriori(lista,
                            min_support = soporte,
                            min_confidence = confianza,
                            min_lift = elevacion)
            ResultadoC1 = list(ReglasC1)

            for item in ResultadoC1:
                Emparejar = item[0]
                items = [x for x in Emparejar]
                Reglas.append(str(item[0])[11:-2])
                Soporte.append(str(item[1])[:6])
                Confianza.append(str(item[2][0][2])[:6])
                Lift.append(str(item[2][0][3])[:6])

            df = pd.DataFrame({"Reglas": Reglas, 
                               "Soporte": Soporte,
                               "Confianza": Confianza,
                               "Lift": Lift})
            st.markdown("""
                    <style>
                    table td:nth-child(1) {
                        display: none
                    }
                    table th:nth-child(1) {
                        display: none
                    }
                    </style>
                    """, unsafe_allow_html=True)
            st.table(df)  
        else:
            st.write('El algoritmo espero por ti')
        


        if st.sidebar.button("Grafica frecuencia de datos"):
            Grafica(ListaM)

        if st.sidebar.button("Ver datos"):
            st.subheader('Primeros 10 datos\n ')
            st.write(Datos.head(10))


def Grafica(ListaM):
    #Se genera un gráfico de barras
    fig, ax = plt.subplots(figsize=(16,20), dpi=300)
    ax.set_ylabel('Item')
    ax.set_xlabel('Frecuencia')
    ax.barh(ListaM['Item'], width=ListaM['Frecuencia'], color='gold')
    st.write("Grafica de la frecuencia de cada dato")
    st.pyplot(fig)
    
        

    
    
    