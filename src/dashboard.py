#--------- LIBRERIAS ---------#

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

#----------- GESTIÓN DE LOS DATOS ----------#

nltk.download('stopwords')

# Carga de datos
data = pd.read_excel(r"data_1.xlsx")
data['FECHA'] = pd.to_datetime(data['FECHA'])

# Creación de columnas adicionales
data['POS_COMENTARIOS'] = data['COMENTARIO_POSITIVO'].notna().astype(int)
data['NEG_COMENTARIOS'] = data['COMENTARIO_NEGATIVO'].notna().astype(int)
data['TOTAL_COMENTARIOS'] = data['POS_COMENTARIOS'] + data['NEG_COMENTARIOS']

# stopwords en español
stopwords_spanish = set(stopwords.words('spanish'))

# Generar la nube de palabras sin iniciar la GUI de Matplotlib
def generate_wordcloud(text, colormap='coolwarm'):
    valid_text = text.dropna().astype(str)
    if valid_text.empty:
        return ""
    
    wordcloud = WordCloud(
        width=600,  # Tamaño ajustado
        height=300, 
        background_color='white', 
        colormap=colormap,
        max_words=100,
        stopwords=stopwords_spanish,
        contour_width=1, 
        contour_color='steelblue',
        scale=1.5  # Ajuste de escala para una mejor calidad
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
CALIFICACION_OPTIONS = [
    {'label': '1 Estrella', 'value': 1},
    {'label': '2 Estrellas', 'value': 2},
    {'label': '3 Estrellas', 'value': 3},
    {'label': '4 Estrellas', 'value': 4},
    {'label': '5 Estrellas', 'value': 5}
]

# ----------------- LAYOUT -----------------#
app.layout = dbc.Container([
    # Header Section
    dbc.Row([
        dbc.Col(html.H1("Dashboard", style={'textAlign': 'center', 'padding': '10px', 'color': '#fff'}), width=8),
        dbc.Col([
            html.Label('Filtrar por Tipo de Habitación:', style={'fontWeight': 'bold', 'font-size': '14px'}),
            dcc.Dropdown(
                id='filtro-tipo-hab',
                options=[{'label': str(tipo), 'value': str(tipo)} for tipo in sorted(data['TIPO_HAB'].dropna().astype(str).unique())],
                multi=True,
                placeholder='Selecciona tipo de habitación...',
            )
        ], width=2),
        dbc.Col([
            html.Label('Filtrar por Número de Habitación:', style={'fontWeight': 'bold', 'font-size': '14px'}),
            dcc.Dropdown(
                id='filtro-num-hab',
                options=[{'label': str(num), 'value': str(num)} for num in sorted(data['No_HAB'].dropna().astype(str).unique())],
                multi=True,
                placeholder='Selecciona número de habitación...'
            )
        ], width=2),
    ], style={'backgroundColor': '#2c3e50', 'height': '10vh', 'padding': '10px'}),
    
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
            
        ], width=2, style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'height': '80vh', 'overflowY': 'auto'}), 

        # Área de visualizaciones dividida en dos columnas
        dbc.Col([
            dbc.Row([
                # Columna izquierda, dividida en dos filas (Nube de Palabras y Tabla de Comentarios)
                dbc.Col([
                    # Nube de Palabras
                    dbc.Card([
                        html.H4('Nube de Palabras', style={'textAlign': 'center', 'color': '#333', 'font-size': '16px'}),
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
                ], width=6),

                # Columna derecha con pestañas de visualización
                dbc.Col([
                    dbc.Card([
                        dbc.Tabs([
                            dbc.Tab(label='Línea de Reservas', tab_id='tab-linea-reservas'),
                            dbc.Tab(label='Gráfico de Barras por Tipo de Habitación', tab_id='tab-barras-hab'),
                            dbc.Tab(label='Mapa de Comentarios por País', tab_id='tab-mapa-comentarios')
                        ], id='tabs-comentarios', active_tab='tab-linea-reservas'),
                        dcc.Loading(
                            children=html.Div(id='contenido-tab-comentarios', style={'height': '75vh', 'padding': '10px'}),
                            type="circle"
                        )
                    ], style={'height': '80vh', 'border': 'none'})  
                ], width=6)
            ])
        ], width=10)
    ]),
    
    # Footer Section
    dbc.Row([ 
        dbc.Col(html.P("Proyecto Dashboard.", 
                       style={'textAlign': 'center', 'padding': '10px', 'color': '#fff', 'font-size': '12px'}),
                width=12)
    ], style={'backgroundColor': '#2c3e50', 'height': '10vh', 'padding': '0px'})

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
        filtered_data = filtered_data[filtered_data['CALIFICACION'].isin(calificacion_value)]
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
    nube_palabras = 'data:image/png;base64,{}'.format(generate_wordcloud(selected_text, 'coolwarm'))

    if active_tab == 'tab-linea-reservas':
        fig_lineas = go.Figure()
        if 'positivo' in selected_comments:
            reservas_positivas = filtered_data.groupby('FECHA')['POS_COMENTARIOS'].sum().resample(agg_type).sum()
            fig_lineas.add_trace(go.Scatter(x=reservas_positivas.index, y=reservas_positivas, 
                                            mode='lines', name='Positivos', line=dict(color='green')))
        if 'negativo' in selected_comments:
            reservas_negativas = filtered_data.groupby('FECHA')['NEG_COMENTARIOS'].sum().resample(agg_type).sum()
            fig_lineas.add_trace(go.Scatter(x=reservas_negativas.index, y=reservas_negativas, 
                                            mode='lines', name='Negativos', line=dict(color='red')))
        contenido_tab_comentarios = dcc.Graph(figure=fig_lineas)

    elif active_tab == 'tab-barras-hab':
        tipo_hab_data = filtered_data['TIPO_HAB'].value_counts().reset_index()
        tipo_hab_data.columns = ['Tipo de Habitación', 'Cantidad']
        fig_barras = px.bar(tipo_hab_data, x='Tipo de Habitación', y='Cantidad', title='Comentarios por Tipo de Habitación')
        contenido_tab_comentarios = dcc.Graph(figure=fig_barras)

    elif active_tab == 'tab-mapa-comentarios':
        mapa_data = filtered_data[['PAIS', 'POS_COMENTARIOS', 'NEG_COMENTARIOS']].groupby('PAIS').sum().reset_index()
        mapa_fig = px.scatter_geo(mapa_data, locations='PAIS', locationmode='country names', 
                                  size='POS_COMENTARIOS' if 'positivo' in selected_comments else 'NEG_COMENTARIOS',
                                  projection='natural earth', title='Distribución de Comentarios por País')
        contenido_tab_comentarios = dcc.Graph(figure=mapa_fig)

    return opciones_palabras, palabras_clave_seleccionadas, '', nube_palabras, contenido_tab_comentarios, tabla_data, tabla_columns

if __name__ == '__main__':
    app.run_server(debug=True)
