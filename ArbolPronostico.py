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

def impresion():
    col1, col2, col3 = st.columns(3)
    with col2:
        st.title("Arboles de pronostico")
        imagen = im.open('Imagenes\ArbolPro.jpg')
        st.image(imagen, caption = 'Arbol pronostico')