import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np
from PIL import Image as im
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import Modulo

def impresion():
    aplicado = 0
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.title("Arboles de Clasificacion")
        imagen = im.open('Imagenes\ArbolDeci.jpg')
        st.image(imagen, caption = 'Arbol de Decision')
    #--------------------Lectura de datos--------------------------------------
    st.subheader('Elige el archivo con los datos a trabajar para iniciar\n ')
    Datos_subidos = st.file_uploader(" ", type = 'csv')
    
    if Datos_subidos is not None:
    #-------------------DATOS------------------------------------------
        Datos = pd.read_csv(Datos_subidos)
        Encabezado = list(Datos.columns)
    #------------------GRAFICA-----------------------------------------
        with st.expander("Graficador"):
            with st.form("Grafica"):
                st.header('Grafica')
                variableY = st.selectbox(
                'Elige la variable para el eje Y',
                Encabezado)
                MatrizY = np.array(Datos[variableY])

                variableX = st.selectbox(
                'Elige la variable para el eje X',
                Encabezado)
                MatrizX = np.array(Datos[variableX])

                submitted = st.form_submit_button("Graficar")
                if submitted:
                    with st.spinner('Graficando.... esto puede demorar unos segundos'):
                        Grafica(Datos,variableY, variableX, MatrizY, MatrizX)
    #----------------------MAPA DE CALOR---------------------------------
        with st.expander("Mapa calor"):
            with st.spinner('Cargando el mapa de calor...'):
                fig, ax = plt.subplots(figsize=(14,10), dpi=200)
                MatrizInf = np.triu(Datos.corr())
                sns.heatmap(Datos.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
                st.pyplot(fig)

    #-----------------------SELECCION DE VARIABLES-------------------------
        with st.form("my_form"):
            st.header('Seleccion de variables predictoras')
            predictoras = st.multiselect('Selecciona las variables predictoras para el algoritmo',
                                    Encabezado)
            MatrizP = np.array(Datos[predictoras])

            st.header('Seleccion de variable clase')
            vclase = st.selectbox('Selecciona la variable de clase',
                                    Encabezado)
            MatrizC = np.array(Datos[vclase])
            
            st.header('Seleccion de parametros para el algoritmo')
            col1, col2 = st.columns(2)
            with col1:
                t_s = st.number_input('Test size', min_value = 0.01, max_value = 0.99, value = 0.2)
                r_s = st.number_input('Random state', min_value = 0, max_value = 9999, step = 1, value = 1234)
                m_d = st.number_input('Max Depth', min_value = 1, max_value = 100, step = 1, value = 8)
            with col2: 
                m_s_s = st.number_input('Min Samples Split', min_value = 2, max_value = 100, step = 1, value = 4)
                m_s_l = st.number_input('Min Samples Leaf', min_value = 1, max_value = 100, step = 1, value = 2)
            submitted = st.form_submit_button("Aplicar algoritmo")
            if submitted or len(predictoras) != 0:
                if len(predictoras)== 0 or len(vclase) == 0:
                    aplicado = 2
                else:
                    aplicado = 1
                    with st.expander('Matriz variables predictoras'):
                        st.write(pd.DataFrame(MatrizP))
                    with st.expander('Matriz variable de clase'):
                        st.write(pd.DataFrame(MatrizC))
    #--------------DIVISION DE DATOS------------------------------------
        if aplicado == 1:
            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(MatrizP, MatrizC, 
                                                                                    test_size = t_s, 
                                                                                    random_state = r_s,
                                                                                    shuffle = True)
    
    #---------------ENTRENAMOS AL MODELO---------------------------------
            ClasificacionAD = DecisionTreeClassifier(max_depth=m_d, min_samples_split=m_s_s, min_samples_leaf=m_s_l)
            ClasificacionAD.fit(X_train, Y_train)
    
    #-----------------Clasificacion-----------------------------------------
            Y_Clasificacion = ClasificacionAD.predict(X_validation)
            Valores = pd.DataFrame({
                "Reales": Y_validation, 
                "Clasificacion": Y_Clasificacion})
            with st.expander('Clasificacion'):
                st.subheader('Valores reales VS  clasificados')
                st.write(Valores)
                with st.spinner('Graficando.... esto puede demorar unos segundos'):
                    fig, ax = plt.subplots(figsize=(30,10), dpi=300)
                    ax.plot(Y_validation, color='green', marker='o', label='Y_validation')
                    ax.plot(Y_Clasificacion, color='red', marker='o', label='Y_Clasificacion')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
    
    #------------------DATOS DEL MODELO---------------------------------
            with st.expander('Datos del modelo'):
                efectividad = ClasificacionAD.score(X_validation, Y_validation)
                st.info('La efectividad del modelo es del '+str(efectividad*100)[:5]+'%')
                #st.write(palabras)
                
                Y_Clasificacion = ClasificacionAD.predict(X_validation)
                Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                            Y_Clasificacion, 
                                            rownames=['R'], 
                                            colnames=['C']) 
                palabras = str(Matriz_Clasificacion).split()
                st.subheader('Matriz de clasificacion')
                st.write(Matriz_Clasificacion)
                st.write('Se obtuvieron '+palabras[5]+
                            ' clasificados como '+palabras[1]+
                            ' correctamente  y '+ palabras[9]+ 
                            ' clasificados como '+palabras[2]+
                            ' correctamente. Se obtuvieron '+palabras[8]+
                            ' clasificados como '+palabras[1]+
                            ' que en realidad eran '+ palabras[2]+
                            ' y se obtuvieron '+palabras[6] +
                            ' clasificados como '+ palabras[2]+
                            ' que en realidad eran '+palabras[1])

                st.subheader('ARBOL RESULTANTE')
                with st.spinner('Graficando.... esto puede demorar unos segundos'):
                    fig, ax = plt.subplots(figsize=(30,15), dpi=300)
                    plot_tree(ClasificacionAD, feature_names = predictoras)
                    st.pyplot(fig)
    #-------------------USAR MODELO--------------------------------------
            with st.expander('Usar modelo'):
                DatosUsuario = { }
                with st.form('Usando'):
                    st.subheader('Seleccion de valores de las variables')
                    col1, col2, col3 = st.columns(3)
                    for i in range(0,len(predictoras),3):
                        with col1:
                            DatosUsuario[predictoras[i]] = [st.number_input('Inserte el campo '+ predictoras[i], key = 1,format = '%f')]
                        with col2:
                            if i+1 < len(predictoras):
                                DatosUsuario[predictoras[i+1]] = [st.number_input('Inserte el campo '+ predictoras[i+1], key = 2,format = '%f')]
                        with col3: 
                            if i+2 < len(predictoras):
                                DatosUsuario[predictoras[i+2]] = [st.number_input('Inserte el campo '+ predictoras[i+2], key = 3,format = '%f')]
                    submitted = st.form_submit_button("Aplicar modelo")
                    if submitted:
                        UsuarioNuevo = pd.DataFrame(DatosUsuario)
                        st.info('El Clasificacion del nuevo usuario es de '+ str(ClasificacionAD.predict(UsuarioNuevo))[1:-1])
            
        elif aplicado == 2:
            st.header('Llene las variables predictoras y de clase')    
    #------------------------BOTONES SIDEBAR---------------------------
        if st.sidebar.button("Ver datos"):
            st.subheader('Primeros 10 datos\n ')
            st.write(Datos.head(10))

def Grafica(Datos, variableY, variableX, MatrizY, MatrizX):
    #Se genera un gráfico de barras
    fig, ax = plt.subplots(figsize=(30,10), dpi=300)
    ax.set_ylabel(variableX)
    ax.set_xlabel(variableY)
    ax.plot(MatrizX, MatrizY, color='green', marker='o')
    ax.legend()
    st.pyplot(fig)