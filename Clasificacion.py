import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np
from PIL import Image as im
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

def impresion():
    aplicado = 0
    tipoGraf = 0
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.title("Clasificacion con regresion logistica")
        imagen = im.open('Imagenes\Clasificacion.jpg')
        st.image(imagen, caption = 'Clasificacion', width = 500)

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
                fig, ax = plt.subplots(figsize=(14,10), dpi=200)
                MatrizInf = np.triu(CorrDatos)
                sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
                st.pyplot(fig)
        else:
            pass
    
    #----------------SELECCION DE VARIABLES----------------------------
        with st.form("my_form"):
            st.header('Seleccion de variables predictoras')
            predictoras = st.multiselect('Selecciona las variables para el algoritmo',
                                    Encabezado, Encabezado[3])
            MatrizP = np.array(Datos[predictoras])

            st.header('Seleccion de variable clase')
            vclase = st.selectbox('Selecciona la variable de clase',
                                    Encabezado)
            MatrizC = np.array(Datos[vclase])
            
            st.header('Seleccion de parametros para el algoritmo')
            t_s = st.number_input('Test size', min_value = 0.01, max_value = 0.99)
            r_s = st.number_input('Random state', min_value = 0, max_value = 9999, step = 1)
            submitted = st.form_submit_button("Aplicar algoritmo")
            if submitted or len(predictoras) != 0:
                aplicado = 1
                with st.expander('Matriz variables predictoras'):
                    st.write(pd.DataFrame(MatrizP))
                with st.expander('Matriz variable de clase'):
                    st.write(pd.DataFrame(MatrizC))
    #--------------DIVISION DE DATOS------------------------------------
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(MatrizP, MatrizC, 
                                                                                test_size = t_s, 
                                                                                random_state = r_s,
                                                                                shuffle = True)
    #---------------------------ENTRENAMOS AL MODELO----------------------
        Clasificacion = linear_model.LogisticRegression()
        Clasificacion.fit(X_train, Y_train)
        Probabilidad = Clasificacion.predict_proba(X_validation)
        Predicciones = Clasificacion.predict(X_validation)
    #---------------------------VALIDACION DEL MODELO---------------------
        exactitud = Clasificacion.score(X_validation, Y_validation)
        Y_Clasificacion = Clasificacion.predict(X_validation)
        Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                            Y_Clasificacion, 
                                            rownames=['R'], 
                                            colnames=['C']) 
        with st.expander('Datos del modelo'):
            if aplicado == 1:
                palabras = str(Matriz_Clasificacion).split()
                #st.write(palabras)
                st.subheader('La exactitud del modelo es de '+str(exactitud * 100)+ '%')
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
            else:
                st.subheader('Aplica el algoritmo antes de usar el modelo')

        with st.expander('Usar modelo'):
            if aplicado == 1:
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
                        st.write('El usuario nuevo ha sido clasificado como '+ str(Clasificacion.predict(UsuarioNuevo))[1:-1])
            else:
                st.subheader('Aplica el algoritmo antes de usar el modelo')
            
    #------------------------BOTONES SIDEBAR---------------------------
        if st.sidebar.button("Ver datos"):
            st.subheader('Primeros 10 datos\n ')
            st.write(Datos.head(10))
#M = 0 | B = 1
            



