import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy as np
from PIL import Image as im

def entrada():
    st.title("Proyecto de Inteligencia artificial")
    st.write("### **Alumno: Jesús Bryan parada Pérez**")
    imagen = im.open('Imagenes\Imagen_de_Inicio.jpg')
    st.image(imagen, caption = 'Talos principle')



