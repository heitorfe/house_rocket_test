import geopandas
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px

st.set_page_config(layout = 'wide')

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

def set_feature(data):
    # add new feature
    data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.0929)
    return data

def overview_data(data):
    # ==================
    # Data Overview
    # ==================

    f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect(
        'Enter zipcode',
        data['zipcode'].unique())

    st.write(f_attributes)
    st.write(f_zipcode)

    st.title('Data Overview')

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]

    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]

    elif (f_zipcode == []) & (f_attributes == []):
        data = data.copy()
    st.dataframe(data)
    c1, c2 = st.columns((2, 1))
    # Average metrics

    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQFT LIVING', 'PRICE/M2']

    c1.header('Average Values')
    c1.dataframe(df, height=800)

    # Statistic Descriptive

    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    median = pd.DataFrame(num_attributes.apply(np.median))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))
    std = pd.DataFrame(num_attributes.apply(np.std))

    df1 = pd.concat([max_, min_, media, median, std], axis=1).reset_index()

    df1.columns = ['ATTRIBUTES', 'MAX', 'MIN', 'MEDIA', 'MEDIANA', 'STD']

    c2.header('Descriptive Analysis')
    c2.dataframe(df1, height=800)
    return None

def portifolio_density(data, geofile):
    # =======================
    # Densidade de portifolio
    # =======================

    st.title('Region Overview')

    c1, c2 = st.columns((2, 3))
    c1.header('Portifolio Density')

    df = data.sample(10)

    # Base map - Folium
    density_map = folium.Map(location=[data['lat'].mean(),
                                       data['long'].mean()],
                             default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R${0} on: {1}. Features: {2} Sqft,'
                            ' {3} bedrooms, {4} bathrooms, {5} year built '.format(
                          row['price'],
                          row['date'],
                          row['sqft_living'],
                          row['bedrooms'],
                          row['bathrooms'],
                          row['yr_built'])).add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    # Region Price Map

    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()

    df.columns = ['ZIP', 'PRICE']

    # df = df.sample(10)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(),
                                            data['long'].mean()],
                                  default_zoom_start=15)

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE')

    with c2:
        folium_static(region_price_map)
    return None

def commercial_distribution(data):
    # ================================================
    # Distribuição dos imóveis por categoria comercial
    # ================================================

    st.sidebar.title('Commercial Options')
    st.title('Comercial Attributes')

    # ------------- Average Price per Year

    # filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built',
                                     min_year_built,
                                     max_year_built,
                                     min_year_built)

    st.header('Average Price per Year Built')

    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    st.write(f_year_built)

    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ------------- Average Price per Day
    st.header('Average Price per Day')
    st.sidebar.subheader('Select Max Date')

    # filters
    data['date'] = pd.to_datetime(data['date']).dt.date

    min_date = data['date'].min()
    max_date = data['date'].max()

    f_date = st.sidebar.slider(
        label='Minimum Date',
        min_value=min_date,
        max_value=max_date,
        value=min_date
    )

    st.write(f_date)

    # data filtering
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # -------------- Histograma
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    # filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]

    # data plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)
    return None

def attributes_distribution(data):
    # ================================================
    # Distribuição dos imoveis por categorias físicas
    # ================================================

    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    # filters
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms',
                                      data['bedrooms'].sort_values().unique())

    f_bathrooms = st.sidebar.selectbox('Max number of bedrooms',
                                       data['bathrooms'].sort_values().unique())

    c1, c2 = st.columns(2)

    # House per bedrooms
    c1.header('Houses per bedrooms')
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per bathrooms

    c2.header('Houses per bathrooms')
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    # filters
    f_floors = st.sidebar.selectbox('Max number of Floors', data['floors'].sort_values().unique())
    f_waterfront = st.sidebar.checkbox('Waterview', False)

    c1, c2 = st.columns(2)

    # House per floors
    c1.header('Houses per floor')
    df = data[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per waterview
    c2.header('Houses with Waterview')
    if f_waterfront:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)
    return None

if __name__ == '__main__':

    #data extraction------------------------

    #get data
    path = 'kc_house_data.csv'
    data = get_data(path)

    #get geofile
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    geofile = get_geofile(url)

    #transformation---------------------------

    data = set_feature(data)

    overview_data(data)

    portifolio_density(data, geofile)

    commercial_distribution(data)

    attributes_distribution(data)

    #loading--------------------------------