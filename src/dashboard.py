import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Cambia el backend a un modo sin GUI
import base64
from io import BytesIO
import nltk
from nltk.corpus import stopwords
import time
import os
import pickle

#----------- GESTIÓN DE LOS DATOS ----------#

nltk.download('stopwords')

# Carga de datos
print("Cargando datos...")
# Define la ruta absoluta al archivo
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data", "data_1.xlsx")

# Carga el archivo de datos
data = pd.read_excel(data_path)
data['FECHA'] = pd.to_datetime(data['FECHA'])

# Asegurarse de que la columna 'CALIFICACION' sea de tipo numérico
print("Convirtiendo calificación a tipo numérico...")
data['CALIFICACION'] = pd.to_numeric(data['CALIFICACION'], errors='coerce')

# Creación de columnas adicionales
print("Creando columnas adicionales de comentarios...")
data['POS_COMENTARIOS'] = data['COMENTARIO_POSITIVO'].notna().astype(int)
data['NEG_COMENTARIOS'] = data['COMENTARIO_NEGATIVO'].notna().astype(int)
data['TOTAL_COMENTARIOS'] = data['POS_COMENTARIOS'] + data['NEG_COMENTARIOS']

# stopwords en español
print("Descargando stopwords en español...")
stopwords_spanish = set(stopwords.words('spanish'))

# Generar la nube de palabras sin iniciar la GUI de Matplotlib
def generate_wordcloud(text, colormap='Purples'):
    print("Generando nube de palabras...")
    valid_text = text.dropna().astype(str)
    if valid_text.empty:
        print("Texto vacío para la nube de palabras. Retornando cadena vacía...")
        return ""
    
    wordcloud = WordCloud(
        width=600,
        height=300,
        background_color='white',
        colormap='Purples',
        max_words=100,
        stopwords=stopwords_spanish,
        contour_width=1.0,
        contour_color='black',
        scale=1.5,
        normalize_plurals=False
    ).generate(' '.join(valid_text))
    
    buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

#-------Inicialización de la aplicación Dash con Bootstrap-------#
print("Inicializando la aplicación Dash...")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True
server = app.server

# Opciones de agrupación (desplegable)
AGG_OPTIONS = {
    'D': 'Día',
    'W': 'Semana',
    'ME': 'Mes',
    'Q': 'Trimestre',
    'Y': 'Año'
}

# Lista de opciones de calificación
CALIFICACION_OPTIONS = [{'label': f'{i}-{i+1}', 'value': f'{i}-{i+1}'} for i in range(1, 10)]

# ----------------- LAYOUT -----------------#
print("Configurando el layout de la aplicación...")
app.layout = dbc.Container([
    # Header Section
    dbc.Row([
        dbc.Col(html.H1("ANÁLISIS - RESEÑAS DE ALOJAMIENTO R10", style={'textAlign': 'left', 'padding': '10px', 'color': '#fff','fontSize': '28px'}), xs=12, sm=12, md=4, lg=4),
        dbc.Col([
            html.Label('Filtrar por Tipo de Habitación:', style={'fontWeight': 'bold', 'font-size': '14px', 'color': '#fff'}),
            dcc.Dropdown(
                id='filtro-tipo-hab',
                options=[{'label': str(tipo), 'value': str(tipo)} for tipo in sorted(data['TIPO_HAB'].dropna().astype(str).unique())],
                multi=True,
                placeholder='Selecciona tipo de habitación...'
            )
        ], xs=12, sm=12, md=4, lg=4),
        dbc.Col([
            html.Label('Filtrar por Número de Habitación:', style={'fontWeight': 'bold', 'font-size': '14px', 'color': '#fff'}),
            dcc.Dropdown(
                id='filtro-num-hab',
                options=[{'label': str(num), 'value': str(num)} for num in sorted(data['No_HAB'].dropna().astype(str).unique())],
                multi=True,
                placeholder='Selecciona número de habitación...'
            )
        ], xs=12, sm=12, md=4, lg=4),
    ], style={'backgroundColor': '#2c3e50', 'padding': '10px'}),
    
    dbc.Row([
        # Barra lateral de filtros
        dbc.Col([
            html.H2('Filtros', style={'textAlign': 'center', 'font-size': '20px'}),
            html.Div([
                dcc.Checklist(
                    id='comentarios-tipo',
                    options=[
                        {'label': 'Comentarios Positivos', 'value': 'positivo'},
                        {'label': 'Comentarios Negativos', 'value': 'negativo'}
                    ],
                    value=['positivo', 'negativo'],
                    inline=False,
                    style={'marginBottom': '10px', 'font-size': '14px'}
                )
            ], style={'marginBottom': '15px', 'marginLeft': '10px', 'marginRight': '10px'}),  

            html.Div([
                html.Label('Agregación por:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'font-size': '14px'}),
                dcc.Dropdown(
                    id='agg-dropdown',
                    options=[{'label': v, 'value': k} for k, v in AGG_OPTIONS.items()],
                    value='ME',
                    clearable=False,
                    style={'marginBottom': '10px'}
                )
            ], style={'marginBottom': '15px', 'marginLeft': '10px', 'marginRight': '10px'}),
            
            html.Div([
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date='2019-01-01',
                    end_date='2024-12-31',
                    min_date_allowed='2019-01-01',
                    max_date_allowed='2024-12-31',
                    display_format='YYYY-MM-DD',
                    style={'width': '100%', 'marginBottom': '10px'}
                )
            ], style={'marginBottom': '15px', 'marginLeft': '10px', 'marginRight': '10px'}),

            html.Div([
                html.Label('Filtrar por Calificación:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'font-size': '14px'}),
                dcc.Dropdown(
                    id='filtro-calificacion',
                    options=CALIFICACION_OPTIONS,
                    multi=True,
                    placeholder='Selecciona calificación...',
                    style={'width': '100%', 'marginBottom': '10px'}
                )
            ], style={'marginBottom': '15px', 'marginLeft': '10px', 'marginRight': '10px'}),
            
            html.Div([
                html.Label('Buscar palabras clave:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'font-size': '14px'}),
                dcc.Input(
                    id='input-palabra-clave',
                    type='text',
                    placeholder='Escribe una palabra...',
                    debounce=True,
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
                dcc.Dropdown(
                    id='buscador-palabras',
                    options=[],
                    multi=True,
                    placeholder='Selecciona palabras clave...',
                    style={'width': '100%', 'marginBottom': '10px'}
                )
            ], style={'marginBottom': '15px', 'marginLeft': '10px', 'marginRight': '10px'}),

            html.Div([
                html.Label('Filtrar por País:', style={'fontWeight': 'bold', 'font-size': '14px'}),
                dcc.Dropdown(
                    id='filtro-pais',
                    options=[{'label': str(pais), 'value': str(pais)} for pais in sorted(data['PAIS'].dropna().unique())],
                    multi=True,
                    placeholder='Selecciona país(es)...',
                )
            ], style={'marginBottom': '15px', 'marginLeft': '10px', 'marginRight': '10px'}),
            
        ], xs=12, sm=12, md=3, lg=2, style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'height': '90vh', 'overflowY': 'auto'}), 

        # Área de visualizaciones dividida en dos columnas
        dbc.Col([
            dbc.Row([
                # Columna izquierda, dividida en dos filas (Nube de Palabras y Tabla de Comentarios)
                dbc.Col([
                    # Nube de Palabras
                    dbc.Card([
                        dcc.Loading(
                            children=[
                                html.Div(
                                    html.Img(id='nube-palabras', style={'width': '100%', 'height': '100%', 'object-fit': 'contain'}),
                                    style={'width': '100%', 'height': '100%', 'overflow': 'hidden'}
                                )
                            ],
                            type="circle"
                        )
                    ], style={'marginBottom': '20px', 'height': '40vh'}),  # Ajuste de altura para evitar corte

                    # Tabla de Comentarios
                    dbc.Card([
                        html.H4('Comentarios Positivos y Negativos', style={'textAlign': 'center', 'color': '#333', 'font-size': '16px'}),
                        dcc.Loading(
                            children=html.Div([
                                dash.dash_table.DataTable(
                                    id='tabla-comentarios',
                                    style_table={'height': '30vh', 'maxHeight': '30vh', 'overflowY': 'auto'},  # Ajuste de altura para la tabla
                                    style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'fontSize': 12, 'height': 'auto'},
                                    columns=[],
                                    data=[]
                                )
                            ], style={'height': '100%'}),
                            type="circle"
                        )
                    ], style={'height': '30vh'})
                ], xs=12, sm=12, md=6, lg=6),

                # Columna derecha con pestañas de visualización
                dbc.Col([
                    dbc.Card([
                        dbc.Tabs([
                            dbc.Tab(label='Línea de Reservas', tab_id='tab-linea-reservas'),
                            dbc.Tab(label='Gráfico de Barras por Tipo de Habitación', tab_id='tab-barras-hab'),
                            dbc.Tab(label='Gráfico de Barras por Número de Habitación', tab_id='tab-barras-num-hab'),
                            dbc.Tab(label='Distribución de Calificaciones', tab_id='tab-calificaciones'),  # Nueva pestaña agregada
                            dbc.Tab(label='Recuento de Comentarios por País', tab_id='tab-pais-comentarios')  # Nueva pestaña para países
                        ], id='tabs-comentarios', active_tab='tab-linea-reservas'),
                        dcc.Loading(
                            children=html.Div(id='contenido-tab-comentarios', style={'height': '75vh', 'padding': '10px'}),
                            type="circle"
                        )
                    ], style={'height': '80vh', 'border': 'none'})  
                ], xs=12, sm=12, md=6, lg=6)
            ])
        ], xs=12, sm=12, md=9, lg=10)
    ]),
    
# Footer Section
    dbc.Row([ 
        dbc.Col(html.P("Proyecto Dashboard - Residencial 10", 
                       style={'textAlign': 'center', 'padding': '10px', 'color': '#fff', 'font-size': '12px'}),
                width=12)
    ], style={'backgroundColor': '#2c3e50', 'padding': '10px', 'position': 'fixed', 'bottom': '0', 'width': '100%'})

], fluid=True, style={'padding': '0px', 'margin': '0px', 'height': '100vh', 'display': 'flex', 'flex-direction': 'column'})

# ---------------- CALLBACK ----------------
@app.callback(
    [Output('buscador-palabras', 'options'),
     Output('buscador-palabras', 'value'),
     Output('input-palabra-clave', 'value'),
     Output('nube-palabras', 'src'),
     Output('contenido-tab-comentarios', 'children'),
     Output('tabla-comentarios', 'data'),
     Output('tabla-comentarios', 'columns')],
    [Input('input-palabra-clave', 'value'),
     Input('buscador-palabras', 'value'),
     Input('comentarios-tipo', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('agg-dropdown', 'value'),
     Input('filtro-calificacion', 'value'),
     Input('filtro-pais', 'value'),
     Input('filtro-tipo-hab', 'value'),
     Input('filtro-num-hab', 'value'),
     Input('tabs-comentarios', 'active_tab')],
    [State('buscador-palabras', 'options')]
)
def update_visualizations(palabra_clave, palabras_clave_seleccionadas, selected_comments, start_date, end_date, agg_dropdown_value, calificacion_value, paises_seleccionados, tipos_hab_seleccionados, num_hab_seleccionado, active_tab, opciones_palabras):
    agg_type = agg_dropdown_value
    if palabra_clave:
        if {'label': palabra_clave, 'value': palabra_clave} not in opciones_palabras:
            opciones_palabras.append({'label': palabra_clave, 'value': palabra_clave})

    filtered_data = data[(data['FECHA'] >= start_date) & (data['FECHA'] <= end_date)]
    
    if calificacion_value:
        for calificacion_interval in calificacion_value:
            lower, upper = map(int, calificacion_interval.split('-'))
            filtered_data = filtered_data[(filtered_data['CALIFICACION'] >= lower) & (filtered_data['CALIFICACION'] < upper)]
            
    if paises_seleccionados:
        filtered_data = filtered_data[filtered_data['PAIS'].isin(paises_seleccionados)]
    
    if tipos_hab_seleccionados:
        filtered_data = filtered_data[filtered_data['TIPO_HAB'].isin(tipos_hab_seleccionados)]
        
    if num_hab_seleccionado:
        filtered_data = filtered_data[filtered_data['No_HAB'].isin(num_hab_seleccionado)]
        
    if palabras_clave_seleccionadas:
        for palabra in palabras_clave_seleccionadas:
            pos_filtro = filtered_data['COMENTARIO_POSITIVO'].str.contains(rf'\b{palabra}\b', case=False, na=False)
            neg_filtro = filtered_data['COMENTARIO_NEGATIVO'].str.contains(rf'\b{palabra}\b', case=False, na=False)
            filtered_data = filtered_data[(pos_filtro | neg_filtro)]
            filtered_data['COMENTARIO_POSITIVO'] = filtered_data['COMENTARIO_POSITIVO'].where(pos_filtro, None)
            filtered_data['COMENTARIO_NEGATIVO'] = filtered_data['COMENTARIO_NEGATIVO'].where(neg_filtro, None)

    if 'positivo' in selected_comments and 'negativo' not in selected_comments:
        tabla_data = filtered_data[['COMENTARIO_POSITIVO']].dropna().to_dict('records')
        tabla_columns = [{'name': 'Comentario Positivo', 'id': 'COMENTARIO_POSITIVO'}]
    elif 'negativo' in selected_comments and 'positivo' not in selected_comments:
        tabla_data = filtered_data[['COMENTARIO_NEGATIVO']].dropna().to_dict('records')
        tabla_columns = [{'name': 'Comentario Negativo', 'id': 'COMENTARIO_NEGATIVO'}]
    else:
        tabla_data = filtered_data[['COMENTARIO_POSITIVO', 'COMENTARIO_NEGATIVO']].dropna(how='all').to_dict('records')
        tabla_columns = [
            {'name': 'Comentario Positivo', 'id': 'COMENTARIO_POSITIVO'},
            {'name': 'Comentario Negativo', 'id': 'COMENTARIO_NEGATIVO'}
        ]

    selected_text = pd.concat([filtered_data['COMENTARIO_POSITIVO'], filtered_data['COMENTARIO_NEGATIVO']])
    nube_palabras = 'data:image/png;base64,{}'.format(generate_wordcloud(selected_text, 'Purples'))

    if active_tab == 'tab-linea-reservas':
        fig_lineas = go.Figure()
        if 'positivo' in selected_comments:
            reservas_positivas = filtered_data.set_index('FECHA').resample(agg_type)['POS_COMENTARIOS'].sum()
            fig_lineas.add_trace(go.Scatter(x=reservas_positivas.index, y=reservas_positivas, 
                                            mode='lines', name='Positivos', line=dict(color='green')))
        if 'negativo' in selected_comments:
            reservas_negativas = filtered_data.set_index('FECHA').resample(agg_type)['NEG_COMENTARIOS'].sum()
            fig_lineas.add_trace(go.Scatter(x=reservas_negativas.index, y=reservas_negativas, 
                                            mode='lines', name='Negativos', line=dict(color='red')))
        fig_lineas.update_xaxes(tickangle=45, title='Fecha', showgrid=False)
        fig_lineas.update_yaxes(showgrid=False)
        contenido_tab_comentarios = dcc.Graph(figure=fig_lineas)

    elif active_tab == 'tab-barras-hab':
        tipo_hab_data = filtered_data['TIPO_HAB'].value_counts().reset_index()
        tipo_hab_data.columns = ['Tipo de Habitación', 'Cantidad']
        fig_barras = px.bar(tipo_hab_data, x='Tipo de Habitación', y='Cantidad', title='Comentarios por Tipo de Habitación')
        contenido_tab_comentarios = dcc.Graph(figure=fig_barras)

    elif active_tab == 'tab-barras-num-hab':
        num_hab_data = filtered_data['No_HAB'].value_counts().reset_index()
        num_hab_data.columns = ['Número de Habitación', 'Cantidad']
        fig_barras_num = px.bar(num_hab_data, x='Número de Habitación', y='Cantidad', title='Comentarios por Número de Habitación')
        contenido_tab_comentarios = dcc.Graph(figure=fig_barras_num)

    elif active_tab == 'tab-calificaciones':
        calificaciones_data = filtered_data.copy()
        calificaciones_data['Calificación Intervalo'] = pd.cut(calificaciones_data['CALIFICACION'], bins=[0, 2, 4, 6, 8, 10], labels=['1-2', '2-4', '4-6', '6-8', '8-10'])
        calificaciones_grouped = calificaciones_data.groupby([pd.Grouper(key='FECHA', freq=agg_type), 'Calificación Intervalo']).size().reset_index(name='Cantidad')
        calificaciones_grouped['Porcentaje'] = calificaciones_grouped.groupby('FECHA')['Cantidad'].transform(lambda x: 100 * x / x.sum())
        fig_calificaciones = px.bar(
            calificaciones_grouped,
            x='FECHA',
            y='Porcentaje',
            color='Calificación Intervalo',
            title='Distribución de Calificaciones a lo Largo del Tiempo',
            labels={'Porcentaje': 'Porcentaje de Calificaciones'},
            barmode='group'
        )
        fig_calificaciones.update_xaxes(
            showticklabels=False,  # Eliminar etiquetas del eje X
            tickangle=0,
            title='Fecha',
            showgrid=False,
            tickformat='%b %Y' if agg_type in ['ME', 'Q'] else '%Y',
            dtick='M1' if agg_type == 'ME' else ('M3' if agg_type == 'Q' else 'Y1'),
            rangeslider_visible=False,
            tickmode='linear'
        )
        fig_calificaciones.update_yaxes(title='Porcentaje de Calificaciones', showgrid=False)
        fig_calificaciones.update_layout(
            xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
            yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
            bargap=0.15  # Ajuste del espacio entre barras para mejorar la visualización
        )
        contenido_tab_comentarios = dcc.Graph(figure=fig_calificaciones)

    elif active_tab == 'tab-pais-comentarios':
        pais_data = filtered_data.groupby('PAIS').agg({'POS_COMENTARIOS': 'sum', 'NEG_COMENTARIOS': 'sum'}).reset_index()
        pais_data.columns = ['País', 'Comentarios Positivos', 'Comentarios Negativos']
        contenido_tab_comentarios = html.Div([
            dash.dash_table.DataTable(
                id='tabla-pais-comentarios',
                columns=[
                    {'name': 'País', 'id': 'País'},
                    {'name': 'Comentarios Positivos', 'id': 'Comentarios Positivos'},
                    {'name': 'Comentarios Negativos', 'id': 'Comentarios Negativos'}
                ],
                data=pais_data.to_dict('records'),
                style_table={'height': '70vh', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'fontSize': 12, 'height': 'auto'}
            )
        ])

    return opciones_palabras, palabras_clave_seleccionadas, '', nube_palabras, contenido_tab_comentarios, tabla_data, tabla_columns

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
