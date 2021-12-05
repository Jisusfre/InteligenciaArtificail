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
from sklearn import linear_model
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

def MClasificacion(predictoras, Clasificacion):
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
            st.info('El usuario nuevo ha sido clasificado como '+ str(Clasificacion.predict(UsuarioNuevo))[1:-1])

def MArbolPronostico():
    pass

def MArbolDecision():
    pass