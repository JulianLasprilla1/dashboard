import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import nltk
from nltk.corpus import stopwords

# Descargar stopwords de NLTK si es necesario
nltk.download('stopwords')

# Carga de datos
data = pd.read_excel(r"data/data_1.xlsx")

# Asegurarse de que la columna 'FECHA' sea de tipo datetime
data['FECHA'] = pd.to_datetime(data['FECHA'])

# Creación de columnas adicionales
data['POS_COMENTARIOS'] = data['COMENTARIO_POSITIVO'].notna().astype(int)
data['NEG_COMENTARIOS'] = data['COMENTARIO_NEGATIVO'].notna().astype(int)
data['TOTAL_COMENTARIOS'] = data['POS_COMENTARIOS'] + data['NEG_COMENTARIOS']

# Lista de stopwords en español de NLTK
stopwords_spanish = set(stopwords.words('spanish'))

# Función para generar la nube de palabras eliminando las stopwords en español
def generate_wordcloud(text, colormap='coolwarm'):
    valid_text = text.dropna().astype(str)
    if valid_text.empty:
        return ""  # Devolver cadena vacía si no hay texto
    
    # Crear la nube de palabras con stopwords en español
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        colormap=colormap,
        max_words=100,
        stopwords=stopwords_spanish,  # Aplicamos stopwords aquí
        contour_width=1, 
        contour_color='steelblue',
        prefer_horizontal=1.0,
        scale=1.5
    ).generate(' '.join(valid_text))
    
    buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

# Inicialización de la aplicación Dash con Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

# Opciones de agrupación (desplegable)
AGG_OPTIONS = {
    'D': 'Día',
    'W': 'Semana',
    'M': 'Mes',
    'Q': 'Trimestre',  # Trimestre añadido
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

# Layout de la aplicación
app.layout = dbc.Container([
    # Header Section (10% de la altura total)
    dbc.Row([
        dbc.Col(html.H1("Dashboard", style={'textAlign': 'center', 'padding': '10px', 'color': '#fff'}), width=12)
    ], style={'backgroundColor': '#2c3e50', 'height': '10vh', 'padding': '0px'}),
    
    dbc.Row([
        # Barra lateral de filtros
        dbc.Col([
            html.H2('Filtros', style={'textAlign': 'center', 'font-size': '20px'}),
            # Checklist para tipos de comentarios
            dcc.Checklist(
                id='comentarios-tipo',
                options=[
                    {'label': 'Comentarios Positivos', 'value': 'positivo'},
                    {'label': 'Comentarios Negativos', 'value': 'negativo'}
                ],
                value=['positivo', 'negativo'],  # Eliminado el valor de comentarios totales
                inline=False,
                style={'marginBottom': '10px', 'font-size': '14px'}
            ),
            # Lista desplegable para agregación
            html.Div([
                html.Label('Agregación por:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'font-size': '14px'}),
                dcc.Dropdown(
                    id='agg-dropdown',
                    options=[{'label': v, 'value': k} for k, v in AGG_OPTIONS.items()],
                    value='M',  # Valor inicial establecido en meses
                    clearable=False,
                    style={'marginBottom': '10px'}
                )
            ], style={'marginBottom': '10px'}),
            # Selector de rango de fechas
            dcc.DatePickerRange(
                id='date-picker',
                start_date='2019-01-01',
                end_date='2024-12-31',
                min_date_allowed='2019-01-01',
                max_date_allowed='2024-12-31',
                display_format='YYYY-MM-DD',
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            # Filtro de calificación
            html.Div([
                html.Label('Filtrar por Calificación:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'font-size': '14px'}),
                dcc.Dropdown(
                    id='filtro-calificacion',
                    options=CALIFICACION_OPTIONS,
                    multi=True,  # Permitir seleccionar múltiples calificaciones
                    placeholder='Selecciona calificación...',
                    style={'width': '100%', 'marginBottom': '10px'}
                )
            ]),
            # Buscador de palabras clave dinámico con opciones eliminables
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
                    options=[],  # Las palabras se ingresan dinámicamente
                    multi=True,  # Permitir seleccionar múltiples palabras
                    placeholder='Selecciona palabras clave...',
                    style={'width': '100%', 'marginBottom': '10px'}
                )
            ]),
        ], width=2, style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'height': '80vh', 'overflowY': 'auto'}),  # Ajuste de altura a 80vh y ancho más pequeño
        
        # Área de visualizaciones dividida en cuatro secciones (sin scroll, ocupando el 80% restante de la altura)
        dbc.Col([
            # Definir un grid con cuatro cuadrantes usando CSS Grid
            html.Div([
                # Cuadrante 1: Nube de Palabras
                dbc.Card([
                    html.H4('Nube de Palabras', style={'textAlign': 'center', 'color': '#333', 'font-size': '16px'}),
                    html.Div(
                        html.Img(id='nube-palabras', style={'width': '100%', 'height': '100%', 'object-fit': 'contain'}),
                        style={'width': '100%', 'height': '100%', 'overflow': 'hidden'}  # Contenedor ajustado para evitar desbordes
                    )
                ], className='grid-item', style={'height': '35vh'}),  # Ajuste de altura a 35vh
                
                # Cuadrante 2: Comentarios en el Tiempo
                dbc.Card([
                    html.H4('Comentarios en el Tiempo', style={'textAlign': 'center', 'color': '#333', 'font-size': '16px'}),
                    dcc.Graph(id='grafico-lineas', style={'width': '100%', 'height': '100%'})
                ], className='grid-item', style={'height': '35vh'}),  # Ajuste de altura a 35vh
                
                # Cuadrante 3: Comentarios Positivos y Negativos en Tablas con Scroll
                dbc.Card([
                    html.H4('Comentarios Positivos y Negativos', style={'textAlign': 'center', 'color': '#333', 'font-size': '16px'}),
                    html.Div([
                        html.Div([  # Para alternar positivos y negativos
                            html.H5(id='titulo-tabla-comentarios', style={'textAlign': 'center', 'color': '#333', 'font-size': '14px'}),
                            dash.dash_table.DataTable(
                                id='tabla-comentarios',
                                style_table={'height': '100%', 'maxHeight': '100%', 'overflowY': 'auto', 'border': '1px solid #333'},  # Scroll y ajuste de altura
                                style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'font-size': '12px'},
                                columns=[],
                                data=[]
                            )
                        ], style={'width': '100%', 'height': '100%', 'overflowY': 'auto'})  # Contenedor ajustado para ocupar todo el espacio vertical con scroll
                    ], style={'height': '100%'})
                ], className='grid-item', style={'height': '35vh', 'overflow': 'hidden'}),  # Ajuste de altura a 35vh

                # Cuadrante 4: Mapa de comentarios por país
                dbc.Card([
                    html.H4('Mapa de Comentarios por País', style={'textAlign': 'center', 'color': '#333', 'font-size': '16px'}),
                    dcc.Graph(id='mapa-comentarios', style={'width': '100%', 'height': '100%'})
                ], className='grid-item', style={'height': '35vh'})  # Ajuste de altura a 35vh
            ], style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 1fr',
                'gridTemplateRows': '1fr 1fr',
                'gap': '20px',
                'height': '80vh',  # Altura ajustada para las visualizaciones
                'padding': '10px'
            })
        ], width=10, style={'padding': '10px', 'height': '80vh'})  # Ancho ajustado a 10 para compensar el ajuste en la barra lateral
    ]),

    # Footer Section (10% de la altura total)
    dbc.Row([ 
        dbc.Col(html.P("Proyecto Dashboard.", 
                       style={'textAlign': 'center', 'padding': '10px', 'color': '#fff', 'font-size': '12px'}),
                width=12)
    ], style={'backgroundColor': '#2c3e50', 'height': '10vh', 'padding': '0px'})

], fluid=True, style={'padding': '0px', 'margin': '0px', 'height': '100vh', 'display': 'flex', 'flex-direction': 'column'})


# Callback para agregar palabras clave al Dropdown y actualizar visualizaciones
@app.callback(
    [Output('buscador-palabras', 'options'),
     Output('buscador-palabras', 'value'),
     Output('input-palabra-clave', 'value'),  # Limpiar el input después de agregar la palabra
     Output('nube-palabras', 'src'),
     Output('grafico-lineas', 'figure'),
     Output('tabla-comentarios', 'data'),
     Output('tabla-comentarios', 'columns'),
     Output('titulo-tabla-comentarios', 'children'),
     Output('mapa-comentarios', 'figure')],
    [Input('input-palabra-clave', 'value'),
     Input('buscador-palabras', 'value'),
     Input('comentarios-tipo', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('agg-dropdown', 'value'),
     Input('filtro-calificacion', 'value')],
    [State('buscador-palabras', 'options')]
)
def update_visualizations(palabra_clave, palabras_clave_seleccionadas, selected_comments, start_date, end_date, agg_dropdown_value, calificacion_value, opciones_palabras):
    agg_type = agg_dropdown_value

    # Agregar la palabra clave al dropdown
    if palabra_clave:
        opciones_palabras.append({'label': palabra_clave, 'value': palabra_clave})

    # Filtrar los datos por el rango de fechas seleccionado
    filtered_data = data[(data['FECHA'] >= start_date) & (data['FECHA'] <= end_date)]

    # Filtrar por calificación
    if calificacion_value:
        filtered_data = filtered_data[filtered_data['CALIFICACION'].isin(calificacion_value)]

    # Aplicar filtro de búsqueda de palabras clave
    if palabras_clave_seleccionadas:
        for palabra in palabras_clave_seleccionadas:
            filtered_data = filtered_data[
                filtered_data['COMENTARIO_POSITIVO'].str.contains(rf'\b{palabra}\b', case=False, na=False) |
                filtered_data['COMENTARIO_NEGATIVO'].str.contains(rf'\b{palabra}\b', case=False, na=False)
            ]

    # Filtrar los comentarios según el tipo seleccionado
    if 'positivo' in selected_comments and 'negativo' not in selected_comments:
        selected_text = filtered_data['COMENTARIO_POSITIVO']
        colormap = 'Blues'
        tabla_data = filtered_data[['COMENTARIO_POSITIVO']].dropna().to_dict('records')
        tabla_columns = [{'name': 'Comentario Positivo', 'id': 'COMENTARIO_POSITIVO'}]
        titulo_tabla = 'Comentarios Positivos'
        mapa_data = filtered_data[['PAIS', 'POS_COMENTARIOS']].groupby('PAIS').sum().reset_index()

    elif 'negativo' in selected_comments and 'positivo' not in selected_comments:
        selected_text = filtered_data['COMENTARIO_NEGATIVO']
        colormap = 'Reds'
        tabla_data = filtered_data[['COMENTARIO_NEGATIVO']].dropna().to_dict('records')
        tabla_columns = [{'name': 'Comentario Negativo', 'id': 'COMENTARIO_NEGATIVO'}]
        titulo_tabla = 'Comentarios Negativos'
        mapa_data = filtered_data[['PAIS', 'NEG_COMENTARIOS']].groupby('PAIS').sum().reset_index()

    else:
        # Mostrar tanto positivos como negativos
        selected_text = pd.concat([filtered_data['COMENTARIO_POSITIVO'], filtered_data['COMENTARIO_NEGATIVO']])
        colormap = 'coolwarm'
        tabla_data = filtered_data[['COMENTARIO_POSITIVO', 'COMENTARIO_NEGATIVO']].dropna(how='all').to_dict('records')
        tabla_columns = [{'name': 'Comentario Positivo', 'id': 'COMENTARIO_POSITIVO'}, {'name': 'Comentario Negativo', 'id': 'COMENTARIO_NEGATIVO'}]
        titulo_tabla = 'Comentarios Positivos y Negativos'
        mapa_data = filtered_data[['PAIS', 'POS_COMENTARIOS', 'NEG_COMENTARIOS']].groupby('PAIS').sum().reset_index()

    # Generar la nube de palabras
    nube_palabras = 'data:image/png;base64,{}'.format(generate_wordcloud(selected_text, colormap))

    # Crear el gráfico de líneas para la cantidad de reservas dinámicamente
    fig_lineas = go.Figure()
    if 'positivo' in selected_comments:
        reservas_positivas = filtered_data.groupby('FECHA')['POS_COMENTARIOS'].sum().resample(agg_type).sum()
        fig_lineas.add_trace(go.Scatter(x=reservas_positivas.index, y=reservas_positivas, 
                                        mode='lines', name='Positivos', line=dict(color='green')))
    if 'negativo' in selected_comments:
        reservas_negativas = filtered_data.groupby('FECHA')['NEG_COMENTARIOS'].sum().resample(agg_type).sum()
        fig_lineas.add_trace(go.Scatter(x=reservas_negativas.index, y=reservas_negativas, 
                                        mode='lines', name='Negativos', line=dict(color='red')))
    if not selected_comments:
        reservas_totales = filtered_data.groupby('FECHA')['TOTAL_COMENTARIOS'].sum().resample(agg_type).sum()
        fig_lineas.add_trace(go.Scatter(x=reservas_totales.index, y=reservas_totales, 
                                        mode='lines', name='Total Reservas', line=dict(color='blue')))

    fig_lineas.update_layout(title='Cantidad de Reservas', xaxis_title='Fecha', yaxis_title='Cantidad de Comentarios')

    if not mapa_data.empty:
        mapa_fig = px.scatter_geo(mapa_data, locations='PAIS', locationmode='country names', size='POS_COMENTARIOS' if 'positivo' in selected_comments else 'NEG_COMENTARIOS',
                                  projection='natural earth', title='Distribución de Comentarios por País')
        mapa_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    else:
        mapa_fig = px.scatter_geo()

    return opciones_palabras, palabras_clave_seleccionadas, '', nube_palabras, fig_lineas, tabla_data, tabla_columns, titulo_tabla, mapa_fig


if __name__ == '__main__':
    app.run_server(debug=True)
