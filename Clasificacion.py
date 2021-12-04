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
from sklearn.metrics import confusion_matrix  #Sacar la matriz de los falsos positivos y esos
from sklearn.metrics import accuracy_score

def impresion():
    tipoGraf = 0
    col1, col2, col3 = st.columns(3)
    with col2:
        st.title("Clasificacion con regresion logistica")
        imagen = im.open('Imagenes\Clasificacion.jpg')
        st.image(imagen, caption = 'Clasificacion')

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
            submitted = st.form_submit_button("Aplicar algoritmo")
            
            if submitted:
                with st.expander('Matriz variables predictoras'):
                    st.write(pd.DataFrame(MatrizP))
                with st.expander('Matriz variable de clase'):
                    st.write(pd.DataFrame(MatrizC))
    #--------------DIVISION DE DATOS------------------------------------
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(MatrizP, MatrizC, 
                                                                                test_size = 0.2, 
                                                                                random_state = 1234,
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
            st.subheader('La exactitud del modelo es de '+str(exactitud * 100)+ '%')
            st.subheader('Matriz de clasificacion')
            st.write(Matriz_Clasificacion)
            st.write('Se obtuvieron '+ str(Matriz_Clasificacion)[23:25]+
                        ' clasificados como '+str(Matriz_Clasificacion)[4:5]+
                        ' correctamente  y '+str(Matriz_Clasificacion)[37:39]+ 
                        ' clasificados como '+str(Matriz_Clasificacion)[8:9]+
                        ' correctamente. Se obtuvieron '+str(Matriz_Clasificacion)[33:35]+
                        ' clasificados como '+str(Matriz_Clasificacion)[4:5]+
                        ' que en realidad eran '+str(Matriz_Clasificacion)[8:9]+
                        ' y se obtuvieron '+str(Matriz_Clasificacion)[27:29]+
                        ' clasificados como '+str(Matriz_Clasificacion)[8:9]+
                        ' que en realidad eran '+str(Matriz_Clasificacion)[4:5])
            



