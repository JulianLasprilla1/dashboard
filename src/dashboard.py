import os
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Modo sin GUI para WordCloud
import base64
from io import BytesIO
import nltk
from nltk.corpus import stopwords

# Descargar stopwords en español (solo la primera vez que se ejecute)
nltk.download('stopwords')

# ----------- RUTA Y FUNCIÓN DE DATOS ACTUALIZADOS (Limpieza y Normalización) ---------- #
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data", "data_1.xlsx")

stopwords_spanish = set(stopwords.words('spanish'))

def cargar_datos_actualizados():
    """
    Lee data_1.xlsx, realiza limpieza mínima y normaliza columnas clave:
      - Limpia nombres de columnas (quita espacios al inicio/final).
      - Parsea FECHA con dayfirst=True; elimina filas con FECHA inválida o anterior a 2019-01-01.
      - Convierte CALIFICACION a numérico (errores → NaN).
      - Genera POS_COMENTARIOS, NEG_COMENTARIOS y TOTAL_COMENTARIOS sin alterar texto original.
      - Elimina duplicados exactos (todas las columnas).
      - Strip en columnas categóricas PAIS y TIPO_HAB (sin convertir NaN en cadena).
    """
    # 1) Leer Excel
    df = pd.read_excel(data_path)

    # 2) Normalizar nombres de columnas: quitar espacios sobrantes
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # 3) Limpiar columnas categóricas: quitar espacios solo donde no sea NaN
    for col in ['PAIS', 'TIPO_HAB']:
        if col in df.columns:
            df[col] = df[col].where(df[col].isna(), df[col].str.strip())

    # 4) Parsear FECHA: dd/mm/aaaa o similar, dayfirst=True; errores → NaT
    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')

    # 5) Eliminar filas donde FECHA sea NaT o anterior a 2019-01-01
    if 'FECHA' in df.columns:
        df = df.dropna(subset=['FECHA'])
        df = df[df['FECHA'] >= pd.to_datetime("2019-01-01")]

    # 6) Convertir CALIFICACION a numérico; errores → NaN
    if 'CALIFICACION' in df.columns:
        df['CALIFICACION'] = pd.to_numeric(df['CALIFICACION'], errors='coerce')

    # 7) Generar columnas de conteo de comentarios (sin convertir texto a cadenas)
    if 'COMENTARIO_POSITIVO' in df.columns:
        df['POS_COMENTARIOS'] = df['COMENTARIO_POSITIVO'].notna().astype(int)
    else:
        df['POS_COMENTARIOS'] = 0
    if 'COMENTARIO_NEGATIVO' in df.columns:
        df['NEG_COMENTARIOS'] = df['COMENTARIO_NEGATIVO'].notna().astype(int)
    else:
        df['NEG_COMENTARIOS'] = 0
    df['TOTAL_COMENTARIOS'] = df['POS_COMENTARIOS'] + df['NEG_COMENTARIOS']

    # 8) Eliminar duplicados exactos
    df = df.drop_duplicates().reset_index(drop=True)

    return df

# Leer inicialmente para determinar fecha máxima válida y fijar fecha mínima en 2019-01-01
df_inicial = cargar_datos_actualizados()
if not df_inicial.empty:
    max_fecha = df_inicial['FECHA'].max().date()
else:
    # Si queda vacío tras limpieza, usar hoy como máxima para evitar errores
    max_fecha = pd.Timestamp.today().date()

min_limit_fecha = pd.to_datetime("2019-01-01").date()

def generate_wordcloud(text, colormap='Purples'):
    """
    Genera un WordCloud a partir de una Serie de texto,
    convierte la imagen a base64 y devuelve el string para <img src="data:image/png;base64,..."/>.
    """
    valid_text = text.dropna().astype(str)
    if valid_text.empty:
        return ""
    wordcloud = WordCloud(
        width=600,
        height=300,
        background_color='white',
        colormap=colormap,
        max_words=100,
        stopwords=stopwords_spanish,
        contour_width=1.0,
        contour_color='black',
        scale=1.5,
        normalize_plurals=False
    ).generate(" ".join(valid_text))
    buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(buffer, format="png", bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

# ------- Inicialización de la aplicación Dash con Bootstrap ------- #
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True
server = app.server

# Opciones de agregación temporal
AGG_OPTIONS = {
    'D': 'Día',
    'W': 'Semana',
    'ME': 'Mes',
    'Q': 'Trimestre',
    'Y': 'Año'
}

# Opciones para filtrar calificación (1-2, 2-3, ..., 9-10)
CALIFICACION_OPTIONS = [
    {'label': f'{i}-{i+1}', 'value': f'{i}-{i+1}'} for i in range(1, 10)
]

# ----------------- LAYOUT ----------------- #
app.layout = dbc.Container([

    # Header Section
    dbc.Row([
        dbc.Col(
            html.H1(
                "ANÁLISIS - RESEÑAS DE ALOJAMIENTO R10",
                style={'textAlign': 'left', 'padding': '10px', 'color': '#fff', 'fontSize': '28px'}
            ),
            xs=12, sm=12, md=4, lg=4
        ),
        dbc.Col([
            html.Label(
                'Filtrar por Tipo de Habitación:',
                style={'fontWeight': 'bold', 'font-size': '14px', 'color': '#fff'}
            ),
            dcc.Dropdown(
                id='filtro-tipo-hab',
                options=[],  # Se cargará dinámicamente en callback
                multi=True,
                placeholder='Selecciona tipo de habitación...'
            )
        ], xs=12, sm=12, md=4, lg=4),
        dbc.Col([
            html.Label(
                'Filtrar por Número de Habitación:',
                style={'fontWeight': 'bold', 'font-size': '14px', 'color': '#fff'}
            ),
            dcc.Dropdown(
                id='filtro-num-hab',
                options=[],  # Se cargará dinámicamente en callback
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
                html.Label(
                    'Agregación por:',
                    style={'fontWeight': 'bold', 'marginBottom': '5px', 'font-size': '14px'}
                ),
                dcc.Dropdown(
                    id='agg-dropdown',
                    options=[{'label': v, 'value': k} for k, v in AGG_OPTIONS.items()],
                    value='ME',
                    clearable=False,
                    style={'marginBottom': '10px'}
                )
            ], style={'marginBottom': '15px', 'marginLeft': '10px', 'marginRight': '10px'}),

            # DatePickerRange con formato en español y portal para que no tape el input
            html.Div([
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=min_limit_fecha,    # Siempre desde 2019-01-01
                    end_date=max_fecha,            # Fecha máxima tras limpieza
                    min_date_allowed=min_limit_fecha,
                    max_date_allowed=max_fecha,
                    display_format='DD/MM/YYYY',   # Formato día/mes/año
                    month_format='MMMM YYYY',      # Nombre completo del mes + año
                    first_day_of_week=1,           # Lunes como primer día de la semana
                    with_portal=True,              # Abre calendario sobre el layout para no tapar el input
                    day_size=39,                   # Ajuste de tamaño de celda de día (opcional)
                    number_of_months_shown=1,      # Un mes visible a la vez
                    style={'width': '100%', 'marginBottom': '10px'}
                )
            ], style={'marginBottom': '15px', 'marginLeft': '10px', 'marginRight': '10px'}),

            html.Div([
                html.Label(
                    'Filtrar por Calificación:',
                    style={'fontWeight': 'bold', 'marginBottom': '5px', 'font-size': '14px'}
                ),
                dcc.Dropdown(
                    id='filtro-calificacion',
                    options=CALIFICACION_OPTIONS,
                    multi=True,
                    placeholder='Selecciona calificación...',
                    style={'width': '100%', 'marginBottom': '10px'}
                )
            ], style={'marginBottom': '15px', 'marginLeft': '10px', 'marginRight': '10px'}),

            html.Div([
                html.Label(
                    'Buscar palabras clave:',
                    style={'fontWeight': 'bold', 'marginBottom': '5px', 'font-size': '14px'}
                ),
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
                html.Label(
                    'Filtrar por País:',
                    style={'fontWeight': 'bold', 'font-size': '14px'}
                ),
                dcc.Dropdown(
                    id='filtro-pais',
                    options=[],  # Se cargará dinámicamente en callback
                    multi=True,
                    placeholder='Selecciona país(es)...',
                )
            ], style={'marginBottom': '15px', 'marginLeft': '10px', 'marginRight': '10px'}),

        ], xs=12, sm=12, md=3, lg=2,
           style={'backgroundColor': '#f9f9f9', 'padding': '10px', 'height': '90vh', 'overflowY': 'auto'}),

        # Área de visualizaciones dividida en dos columnas
        dbc.Col([
            dbc.Row([
                # Columna izquierda: Nube de Palabras + Tabla de Comentarios
                dbc.Col([
                    # Nube de Palabras
                    dbc.Card([
                        dcc.Loading(
                            children=[
                                html.Div(
                                    html.Img(
                                        id='nube-palabras',
                                        style={'width': '100%', 'height': '100%', 'object-fit': 'contain'}
                                    ),
                                    style={'width': '100%', 'height': '100%', 'overflow': 'hidden'}
                                )
                            ],
                            type="circle"
                        )
                    ], style={'marginBottom': '20px', 'height': '40vh'}),

                    # Tabla de Comentarios
                    dbc.Card([
                        html.H4(
                            'Comentarios Positivos y Negativos',
                            style={'textAlign': 'center', 'color': '#333', 'font-size': '16px'}
                        ),
                        dcc.Loading(
                            children=html.Div([
                                dash.dash_table.DataTable(
                                    id='tabla-comentarios',
                                    style_table={'height': '30vh', 'maxHeight': '30vh', 'overflowY': 'auto'},
                                    style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'fontSize': 12, 'height': 'auto'},
                                    columns=[],
                                    data=[]
                                )
                            ], style={'height': '100%'}),
                            type="circle"
                        )
                    ], style={'height': '30vh'})
                ], xs=12, sm=12, md=6, lg=6),

                # Columna derecha: Pestañas de visualización
                dbc.Col([
                    dbc.Card([
                        dbc.Tabs([
                            dbc.Tab(label='Línea de Reservas', tab_id='tab-linea-reservas'),
                            dbc.Tab(label='Gráfico de Barras por Tipo de Habitación', tab_id='tab-barras-hab'),
                            dbc.Tab(label='Gráfico de Barras por Número de Habitación', tab_id='tab-barras-num-hab'),
                            dbc.Tab(label='Distribución de Calificaciones', tab_id='tab-calificaciones'),
                            dbc.Tab(label='Recuento de Comentarios por País', tab_id='tab-pais-comentarios')
                        ], id='tabs-comentarios', active_tab='tab-linea-reservas'),
                        dcc.Loading(
                            children=html.Div(
                                id='contenido-tab-comentarios',
                                style={'height': '75vh', 'padding': '10px'}
                            ),
                            type="circle"
                        )
                    ], style={'height': '80vh', 'border': 'none'})
                ], xs=12, sm=12, md=6, lg=6)
            ])
        ], xs=12, sm=12, md=9, lg=10)
    ]),

    # Footer Section
    dbc.Row([
        dbc.Col(
            html.P(
                "Proyecto Dashboard - Residencial 10",
                style={'textAlign': 'center', 'padding': '10px', 'color': '#fff', 'font-size': '12px'}
            ),
            width=12
        )
    ], style={'backgroundColor': '#2c3e50', 'padding': '10px', 'position': 'fixed', 'bottom': '0', 'width': '100%'})

], fluid=True, style={'padding': '0px', 'margin': '0px', 'height': '100vh', 'display': 'flex', 'flex-direction': 'column'})


# ---------------- CALLBACK: ACTUALIZAR VISUALIZACIONES ---------------- #
@app.callback(
    [
        Output('buscador-palabras', 'options'),
        Output('buscador-palabras', 'value'),
        Output('input-palabra-clave', 'value'),
        Output('nube-palabras', 'src'),
        Output('contenido-tab-comentarios', 'children'),
        Output('tabla-comentarios', 'data'),
        Output('tabla-comentarios', 'columns')
    ],
    [
        Input('input-palabra-clave', 'value'),
        Input('buscador-palabras', 'value'),
        Input('comentarios-tipo', 'value'),
        Input('date-picker', 'start_date'),
        Input('date-picker', 'end_date'),
        Input('agg-dropdown', 'value'),
        Input('filtro-calificacion', 'value'),
        Input('filtro-pais', 'value'),
        Input('filtro-tipo-hab', 'value'),
        Input('filtro-num-hab', 'value'),
        Input('tabs-comentarios', 'active_tab')
    ],
    [State('buscador-palabras', 'options')]
)
def update_visualizations(
    palabra_clave,
    palabras_clave_seleccionadas,
    selected_comments,
    start_date,
    end_date,
    agg_dropdown_value,
    calificacion_value,
    paises_seleccionados,
    tipos_hab_seleccionados,
    num_hab_seleccionado,
    active_tab,
    opciones_palabras
):
    # 1) Releer datos en cada llamada (limpieza/normalización incluida)
    df = cargar_datos_actualizados()

    # 2) Actualizar lista de palabras clave sin alterar texto original
    if palabra_clave:
        elemento = {'label': palabra_clave, 'value': palabra_clave}
        if elemento not in opciones_palabras:
            opciones_palabras.append(elemento)

    # 3) Filtrar por rango de fechas (siempre desde 2019 en adelante)
    filtered_data = df[(df['FECHA'] >= pd.to_datetime(start_date)) & (df['FECHA'] <= pd.to_datetime(end_date))]

    # 4) Filtrar por calificación (manteniendo NaN fuera)
    if calificacion_value:
        mask_cal = pd.Series(False, index=filtered_data.index)
        for intervalo in calificacion_value:
            lower, upper = map(int, intervalo.split('-'))
            mask_cal |= (filtered_data['CALIFICACION'] >= lower) & (filtered_data['CALIFICACION'] < upper)
        filtered_data = filtered_data[mask_cal]

    # 5) Filtrar por país
    if paises_seleccionados:
        filtered_data = filtered_data[filtered_data['PAIS'].isin(paises_seleccionados)]

    # 6) Filtrar por tipo de habitación
    if tipos_hab_seleccionados:
        filtered_data = filtered_data[filtered_data['TIPO_HAB'].isin(tipos_hab_seleccionados)]

    # 7) Filtrar por número de habitación
    if num_hab_seleccionado:
        filtered_data = filtered_data[filtered_data['No_HAB'].astype(str).isin(num_hab_seleccionado)]

    # 8) Filtrar por palabras clave en comentarios
    if palabras_clave_seleccionadas:
        temp2 = pd.DataFrame(columns=filtered_data.columns)
        for palabra in palabras_clave_seleccionadas:
            pos_mask = filtered_data['COMENTARIO_POSITIVO'].str.contains(rf'\b{palabra}\b', case=False, na=False)
            neg_mask = filtered_data['COMENTARIO_NEGATIVO'].str.contains(rf'\b{palabra}\b', case=False, na=False)
            subset = filtered_data[pos_mask | neg_mask].copy()
            # Ocultar solo los textos que no coinciden, sin alterar otras columnas
            subset.loc[~pos_mask, 'COMENTARIO_POSITIVO'] = pd.NA
            subset.loc[~neg_mask, 'COMENTARIO_NEGATIVO'] = pd.NA
            temp2 = pd.concat([temp2, subset], axis=0)
        filtered_data = temp2.drop_duplicates().reset_index(drop=True)

    # 9) Preparar datos para la tabla de comentarios según tipo seleccionado
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

    # 10) Generar y codificar nube de palabras
    selected_text = pd.concat([filtered_data['COMENTARIO_POSITIVO'], filtered_data['COMENTARIO_NEGATIVO']])
    nube_palabras_base64 = generate_wordcloud(selected_text, 'Purples')
    img_src = f"data:image/png;base64,{nube_palabras_base64}"

    # 11) Generar contenido según pestaña activa (lógica original intacta)
    if active_tab == 'tab-linea-reservas':
        fig_lineas = go.Figure()
        if 'positivo' in selected_comments:
            reservas_positivas = filtered_data.set_index('FECHA').resample(agg_dropdown_value)['POS_COMENTARIOS'].sum()
            fig_lineas.add_trace(go.Scatter(
                x=reservas_positivas.index,
                y=reservas_positivas,
                mode='lines',
                name='Positivos',
                line=dict(color='green')
            ))
        if 'negativo' in selected_comments:
            reservas_negativas = filtered_data.set_index('FECHA').resample(agg_dropdown_value)['NEG_COMENTARIOS'].sum()
            fig_lineas.add_trace(go.Scatter(
                x=reservas_negativas.index,
                y=reservas_negativas,
                mode='lines',
                name='Negativos',
                line=dict(color='red')
            ))
        fig_lineas.update_xaxes(tickangle=45, title='Fecha', showgrid=False)
        fig_lineas.update_yaxes(showgrid=False)
        contenido_tab_comentarios = dcc.Graph(figure=fig_lineas)

    elif active_tab == 'tab-barras-hab':
        tipo_hab_data = filtered_data['TIPO_HAB'].value_counts().reset_index()
        tipo_hab_data.columns = ['Tipo de Habitación', 'Cantidad']
        fig_barras = px.bar(
            tipo_hab_data,
            x='Tipo de Habitación',
            y='Cantidad',
            title='Comentarios por Tipo de Habitación'
        )
        contenido_tab_comentarios = dcc.Graph(figure=fig_barras)

    elif active_tab == 'tab-barras-num-hab':
        num_hab_data = filtered_data['No_HAB'].astype(str).value_counts().reset_index()
        num_hab_data.columns = ['Número de Habitación', 'Cantidad']
        fig_barras_num = px.bar(
            num_hab_data,
            x='Número de Habitación',
            y='Cantidad',
            title='Comentarios por Número de Habitación'
        )
        contenido_tab_comentarios = dcc.Graph(figure=fig_barras_num)

    elif active_tab == 'tab-calificaciones':
        calificaciones_data = filtered_data.copy()
        calificaciones_data['Calificación Intervalo'] = pd.cut(
            calificaciones_data['CALIFICACION'],
            bins=[0, 2, 4, 6, 8, 10],
            labels=['1-2', '2-4', '4-6', '6-8', '8-10']
        )
        calificaciones_grouped = calificaciones_data.groupby(
            [pd.Grouper(key='FECHA', freq=agg_dropdown_value), 'Calificación Intervalo']
        ).size().reset_index(name='Cantidad')
        calificaciones_grouped['Porcentaje'] = calificaciones_grouped.groupby('FECHA')['Cantidad'].transform(
            lambda x: 100 * x / x.sum()
        )
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
            showticklabels=False,
            tickangle=0,
            title='Fecha',
            showgrid=False,
            tickformat='%b %Y' if agg_dropdown_value in ['ME', 'Q'] else '%Y',
            dtick='M1' if agg_dropdown_value == 'ME' else ('M3' if agg_dropdown_value == 'Q' else 'Y1'),
            rangeslider_visible=False,
            tickmode='linear'
        )
        fig_calificaciones.update_yaxes(
            title='Porcentaje de Calificaciones',
            showgrid=False
        )
        fig_calificaciones.update_layout(
            xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
            yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
            bargap=0.15
        )
        contenido_tab_comentarios = dcc.Graph(figure=fig_calificaciones)

    elif active_tab == 'tab-pais-comentarios':
        pais_data = filtered_data.groupby('PAIS').agg({
            'POS_COMENTARIOS': 'sum',
            'NEG_COMENTARIOS': 'sum'
        }).reset_index()
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

    return opciones_palabras, palabras_clave_seleccionadas or [], '', \
           img_src, \
           contenido_tab_comentarios, tabla_data, tabla_columns


# ---------------- CALLBACK: ACTUALIZAR DROPDOWNS ---------------- #
@app.callback(
    [
        Output('filtro-tipo-hab', 'options'),
        Output('filtro-num-hab', 'options'),
        Output('filtro-pais', 'options')
    ],
    Input('tabs-comentarios', 'active_tab')
)
def actualizar_dropdowns(_):
    """
    Relee data_1.xlsx (con limpieza/normalización) y actualiza opciones de TIPO_HAB, No_HAB y PAIS.
    """
    df = cargar_datos_actualizados()
    tipos_hab = [
        {'label': str(tipo), 'value': str(tipo)}
        for tipo in sorted(df['TIPO_HAB'].dropna().astype(str).unique())
    ]
    num_hab = [
        {'label': str(num), 'value': str(num)}
        for num in sorted(df['No_HAB'].dropna().astype(str).unique())
    ]
    paises = [
        {'label': str(pais), 'value': str(pais)}
        for pais in sorted(df['PAIS'].dropna().astype(str).unique())
    ]
    return tipos_hab, num_hab, paises


# ---------------- MAIN: EJECUTAR SERVIDOR ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
