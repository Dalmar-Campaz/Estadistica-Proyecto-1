import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Medidas de Posición y Box-Plot",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def generar_dataset_ejemplo(semilla: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    temperatura_base = rng.normal(loc=58, scale=5, size=94)
    temperatura_base = np.clip(temperatura_base, 35, None)
    atipicos_temperatura = np.array([92.0, 95.0, 98.0, 101.0, 104.0, 21.0])
    temperatura = np.concatenate([temperatura_base, atipicos_temperatura])
    rng.shuffle(temperatura)
    consumo = np.clip(rng.normal(loc=350, scale=40, size=100), 150, None)
    consumo[[7, 33, 71]] = [720.0, 785.0, 812.0]
    nodos = rng.choice(["NODO-01", "NODO-02", "NODO-03", "NODO-04"], size=100)
    return pd.DataFrame(
        {
            "registro": np.arange(1, 101),
            "nodo": nodos,
            "temperatura_cpu_c": np.round(temperatura, 1),
            "consumo_energia_w": np.round(consumo, 1),
        }
    )


@st.cache_data
def leer_archivo(contenido: bytes, nombre: str) -> pd.DataFrame:
    if nombre.lower().endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(contenido))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(contenido), encoding="latin-1")
    return pd.read_excel(io.BytesIO(contenido))


def calcular_moda(serie: pd.Series):
    if serie.empty:
        return np.nan, 0
    conteo = serie.value_counts()
    max_freq = conteo.max()
    if max_freq == 1 or (conteo == max_freq).all():
        return np.nan, 0
    modas = conteo[conteo == max_freq].index.tolist()
    return float(sorted(modas)[0]), len(modas)


def calcular_estadisticos(serie: pd.Series) -> dict:
    q1, q2, q3 = serie.quantile([0.25, 0.50, 0.75]).tolist()
    iqr = q3 - q1
    moda, cantidad_modas = calcular_moda(serie)
    return {
        "n": int(serie.count()),
        "media": float(serie.mean()),
        "mediana": float(serie.median()),
        "moda": moda,
        "cantidad_modas": cantidad_modas,
        "varianza": float(serie.var(ddof=1)),
        "desviacion": float(serie.std(ddof=1)),
        "minimo": float(serie.min()),
        "maximo": float(serie.max()),
        "q1": float(q1),
        "q2": float(q2),
        "q3": float(q3),
        "iqr": float(iqr),
        "li": float(q1 - 1.5 * iqr),
        "ls": float(q3 + 1.5 * iqr),
        "li_ext": float(q1 - 3.0 * iqr),
        "ls_ext": float(q3 + 3.0 * iqr),
    }


def clasificar_atipico(valor: float, est: dict) -> str:
    if valor < est["li_ext"] or valor > est["ls_ext"]:
        return "Extremo"
    return "Leve"


def construir_boxplot(serie: pd.Series, est: dict, nombre_columna: str) -> go.Figure:
    dentro = serie[(serie >= est["li"]) & (serie <= est["ls"])]
    atipicos = serie[(serie < est["li"]) | (serie > est["ls"])]
    bigote_inferior = float(dentro.min()) if not dentro.empty else est["q1"]
    bigote_superior = float(dentro.max()) if not dentro.empty else est["q3"]

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            q1=[est["q1"]],
            median=[est["q2"]],
            q3=[est["q3"]],
            lowerfence=[bigote_inferior],
            upperfence=[bigote_superior],
            mean=[est["media"]],
            x=[nombre_columna],
            name="Distribución",
            boxmean=False,
            marker_color="#1f77b4",
            line=dict(width=2),
            fillcolor="rgba(31, 119, 180, 0.35)",
            hoverinfo="y",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[nombre_columna],
            y=[est["media"]],
            mode="markers",
            name="Promedio",
            marker=dict(symbol="diamond", size=14, color="#ff7f0e", line=dict(color="black", width=1.5)),
            hovertemplate="Promedio: %{y:.3f}<extra></extra>",
        )
    )
    referencias = [
        ("Q1", est["q1"], "#2ca02c"),
        ("Mediana (Q2)", est["q2"], "#9467bd"),
        ("Q3", est["q3"], "#2ca02c"),
        ("Bigote inferior (dato mínimo ≥ LI)", bigote_inferior, "#7f7f7f"),
        ("Bigote superior (dato máximo ≤ LS)", bigote_superior, "#7f7f7f"),
    ]
    fig.add_trace(
        go.Scatter(
            x=[nombre_columna] * len(referencias),
            y=[r[1] for r in referencias],
            mode="markers",
            name="Cuartiles y bigotes",
            marker=dict(size=9, color=[r[2] for r in referencias], symbol="line-ew-open", line=dict(width=2)),
            text=[r[0] for r in referencias],
            hovertemplate="%{text}: %{y:.3f}<extra></extra>",
        )
    )
    if not atipicos.empty:
        fig.add_trace(
            go.Scatter(
                x=[nombre_columna] * len(atipicos),
                y=atipicos.values,
                mode="markers",
                name="Valores atípicos",
                marker=dict(size=10, color="#d62728", symbol="circle-open", line=dict(width=2)),
                text=[f"Registro #{i}" for i in atipicos.index],
                hovertemplate="%{text}<br>Valor atípico: %{y:.3f}<extra></extra>",
            )
        )
    fig.add_hline(
        y=est["li"],
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"LI = {est['li']:.2f}",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=est["ls"],
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"LS = {est['ls']:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Diagrama de Cajas de «{nombre_columna}»",
        yaxis_title=nombre_columna,
        template="plotly_white",
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    return fig


def construir_histograma(serie: pd.Series, est: dict, nombre_columna: str, bins: int) -> go.Figure:
    fig = px.histogram(
        x=serie,
        nbins=bins,
        histnorm="probability density",
        marginal="rug",
        opacity=0.75,
        color_discrete_sequence=["#1f77b4"],
        labels={"x": nombre_columna},
    )
    fig.update_traces(
        selector=dict(type="histogram"),
        hovertemplate="Intervalo: %{x}<br>Densidad: %{y:.5f}<extra></extra>",
    )
    desviacion = est["desviacion"]
    if desviacion > 0:
        eje = np.linspace(est["minimo"], est["maximo"], 300)
        densidad = np.exp(-0.5 * ((eje - est["media"]) / desviacion) ** 2) / (desviacion * np.sqrt(2 * np.pi))
        fig.add_trace(
            go.Scatter(
                x=eje,
                y=densidad,
                mode="lines",
                name="Densidad normal teórica",
                line=dict(color="#ff7f0e", width=3),
                hovertemplate="x = %{x:.2f}<br>Densidad: %{y:.5f}<extra></extra>",
                xaxis="x",
                yaxis="y",
            )
        )
    fig.add_vline(x=est["media"], line_dash="solid", line_color="#ff7f0e", annotation_text="Media", annotation_position="top")
    fig.add_vline(x=est["q2"], line_dash="dot", line_color="#9467bd", annotation_text="Mediana", annotation_position="top left")
    fig.add_vline(x=est["li"], line_dash="dash", line_color="#d62728")
    fig.add_vline(x=est["ls"], line_dash="dash", line_color="#d62728")
    fig.update_layout(
        title=f"Histograma de «{nombre_columna}» con curva de densidad",
        xaxis_title=nombre_columna,
        yaxis_title="Densidad",
        template="plotly_white",
        height=560,
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def interpretar_asimetria(est: dict) -> str:
    diferencia = est["media"] - est["mediana"]
    rango = est["maximo"] - est["minimo"]
    if rango == 0 or abs(diferencia) < 0.02 * rango:
        return "La media y la mediana son muy cercanas, lo que sugiere una distribución aproximadamente simétrica."
    if diferencia > 0:
        return (
            "La media es mayor que la mediana, lo que indica una **asimetría positiva** "
            "(cola larga a la derecha): los valores altos elevan el promedio."
        )
    return (
        "La media es menor que la mediana, lo que indica una **asimetría negativa** "
        "(cola larga a la izquierda): los valores bajos reducen el promedio."
    )


st.title("📊 Medidas de Posición y Diagramas de Cajas")
st.caption(
    "Estadística aplicada a la Ingeniería de Sistemas — análisis de temperatura de procesadores, "
    "consumo energético y otras métricas de infraestructura."
)

with st.sidebar:
    st.header("⚙️ Controles")
    archivo = st.file_uploader("Sube tu archivo de datos", type=["csv", "xlsx"], help="Formatos admitidos: .csv y .xlsx")
    usar_ejemplo = st.checkbox(
        "Usar Dataset de Ejemplo de Ingeniería de Sistemas",
        value=archivo is None,
        help="Monitoreo de un centro de datos: temperatura de CPU (°C) y consumo de energía (W), con 100 registros y outliers intencionales.",
    )

df = None
origen = ""

if archivo is not None:
    try:
        df = leer_archivo(archivo.getvalue(), archivo.name)
        origen = f"Archivo cargado: {archivo.name}"
    except Exception as error:
        st.sidebar.error(f"No se pudo leer el archivo: {error}")
        df = None

if df is None and usar_ejemplo:
    df = generar_dataset_ejemplo()
    origen = "Dataset de ejemplo: Monitoreo de temperatura y consumo en un centro de datos"

if df is None:
    st.info("👈 Sube un archivo `.csv` o `.xlsx`, o activa el dataset de ejemplo en la barra lateral para comenzar.")
    st.stop()

columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

if not columnas_numericas:
    st.error("El conjunto de datos no contiene columnas numéricas para analizar.")
    st.stop()

with st.sidebar:
    columna = st.selectbox("Columna numérica a analizar", columnas_numericas)
    percentil_k = st.slider("Percentil personalizado (Pₖ)", min_value=1, max_value=99, value=90, step=1)
    numero_bins = st.slider("Número de intervalos del histograma", min_value=5, max_value=60, value=20, step=1)
    st.markdown("---")
    st.caption(origen)

serie = pd.to_numeric(df[columna], errors="coerce").dropna()

if len(serie) < 2:
    st.error("La columna seleccionada necesita al menos 2 valores numéricos válidos.")
    st.stop()

est = calcular_estadisticos(serie)
valor_percentil = float(serie.quantile(percentil_k / 100))
mascara_atipicos = (serie < est["li"]) | (serie > est["ls"])
serie_atipicos = serie[mascara_atipicos]

tab_resumen, tab_graficos, tab_atipicos = st.tabs(
    ["📋 Resumen de Datos y Medidas", "📈 Gráficos Interactivos", "🚨 Valores Atípicos"]
)

with tab_resumen:
    st.subheader("Vista previa de los datos")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Filas totales", f"{len(df):,}")
    col_b.metric("Columnas", f"{df.shape[1]}")
    col_c.metric("Datos válidos analizados", f"{est['n']:,}")
    st.dataframe(df.head(15), use_container_width=True)

    st.subheader("Tendencia central y dispersión")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Promedio (Media)", f"{est['media']:.3f}")
    m2.metric("Mediana", f"{est['mediana']:.3f}")
    m3.metric("Moda", f"{est['moda']:.3f}" if not np.isnan(est["moda"]) else "N/A")
    m4.metric("Varianza muestral (s²)", f"{est['varianza']:.3f}")
    m5.metric("Desv. estándar muestral (s)", f"{est['desviacion']:.3f}")
    if est["cantidad_modas"] > 1:
        st.caption(f"Se detectaron {est['cantidad_modas']} modas (distribución multimodal); se muestra la menor.")
    elif est["cantidad_modas"] == 0:
        st.caption("No existe moda: ningún valor se repite más que los demás.")

    st.subheader("Medidas de posición")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Q₁ (Percentil 25)", f"{est['q1']:.3f}")
    p2.metric("Q₂ (Percentil 50 / Mediana)", f"{est['q2']:.3f}")
    p3.metric("Q₃ (Percentil 75)", f"{est['q3']:.3f}")
    p4.metric("IQR = Q₃ − Q₁", f"{est['iqr']:.3f}")

    st.markdown(f"**Percentil personalizado seleccionado:** P{percentil_k}")
    pk1, pk2 = st.columns(2)
    pk1.metric(f"P{percentil_k}", f"{valor_percentil:.3f}")
    porcentaje_debajo = float((serie <= valor_percentil).mean() * 100)
    pk2.metric("% de datos ≤ Pₖ", f"{porcentaje_debajo:.1f}%")
    st.info(
        f"El {percentil_k}% de los registros de «{columna}» es menor o igual a **{valor_percentil:.3f}**, "
        f"mientras que el {100 - percentil_k}% restante lo supera."
    )

    st.subheader("Resumen descriptivo completo")
    resumen = pd.DataFrame(
        {
            "Medida": [
                "n",
                "Mínimo",
                "Q1",
                "Mediana",
                "Q3",
                "Máximo",
                "Media",
                "Moda",
                "Varianza muestral",
                "Desviación estándar muestral",
                "IQR",
                "Límite inferior (LI)",
                "Límite superior (LS)",
            ],
            "Valor": [
                est["n"],
                est["minimo"],
                est["q1"],
                est["mediana"],
                est["q3"],
                est["maximo"],
                est["media"],
                "N/A" if np.isnan(est["moda"]) else est["moda"],
                est["varianza"],
                est["desviacion"],
                est["iqr"],
                est["li"],
                est["ls"],
            ],
        }
    )
    resumen["Valor"] = resumen["Valor"].apply(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v)
    st.dataframe(resumen, use_container_width=True, hide_index=True)

with tab_graficos:
    st.subheader("Diagrama de cajas (Box-Plot)")
    st.plotly_chart(construir_boxplot(serie, est, columna), use_container_width=True)
    st.caption(
        "Los bigotes llegan hasta el dato más extremo que aún está dentro de los límites LI y LS; "
        "las líneas rojas discontinuas marcan dichos límites exactos. El rombo naranja representa el promedio."
    )
    st.subheader("Histograma con curva de densidad")
    st.plotly_chart(construir_histograma(serie, est, columna, numero_bins), use_container_width=True)
    st.caption(
        "Las barras muestran la densidad empírica; la curva naranja es la densidad normal teórica "
        "con la media y desviación muestrales, útil para contrastar la forma real de los datos."
    )

with tab_atipicos:
    st.subheader("Límites de detección (método de Tukey)")
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Límite inferior (LI)", f"{est['li']:.3f}", help="LI = Q₁ − 1.5 × IQR")
    l2.metric("Límite superior (LS)", f"{est['ls']:.3f}", help="LS = Q₃ + 1.5 × IQR")
    l3.metric("Total de atípicos", f"{len(serie_atipicos)}")
    l4.metric("% del total", f"{len(serie_atipicos) / est['n'] * 100:.2f}%")

    st.latex(r"LI = Q_1 - 1.5 \times IQR \qquad LS = Q_3 + 1.5 \times IQR")

    if serie_atipicos.empty:
        st.success("No se encontraron valores atípicos: todos los datos están dentro de los límites LI y LS.")
    else:
        tabla_atipicos = df.loc[serie_atipicos.index].copy()
        tabla_atipicos.insert(0, "índice_original", serie_atipicos.index)
        tabla_atipicos["tipo_atípico"] = [clasificar_atipico(v, est) for v in serie_atipicos.values]
        tabla_atipicos["posición"] = np.where(serie_atipicos.values < est["li"], "Bajo LI", "Sobre LS")

        leves = int((tabla_atipicos["tipo_atípico"] == "Leve").sum())
        extremos = int((tabla_atipicos["tipo_atípico"] == "Extremo").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Atípicos leves", leves)
        c2.metric("Atípicos extremos", extremos)
        c3.metric("Bajo LI / Sobre LS", f"{int((serie_atipicos < est['li']).sum())} / {int((serie_atipicos > est['ls']).sum())}")

        st.markdown("**Registros atípicos detectados**")
        st.dataframe(tabla_atipicos, use_container_width=True, hide_index=True)

        csv_atipicos = tabla_atipicos.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar atípicos en CSV",
            data=csv_atipicos,
            file_name=f"atipicos_{columna}.csv",
            mime="text/csv",
        )

    st.subheader("Interpretación")
    porcentaje_atipicos = len(serie_atipicos) / est["n"] * 100
    st.markdown(
        f"""
- Los valores considerados **normales** de «{columna}» se encuentran entre **{est['li']:.3f}** y **{est['ls']:.3f}**.
- El **50% central** de los datos (entre Q₁ y Q₃) está en el rango **{est['q1']:.3f} – {est['q3']:.3f}**, con un IQR de **{est['iqr']:.3f}**.
- Se identificaron **{len(serie_atipicos)}** valores atípicos, equivalentes al **{porcentaje_atipicos:.2f}%** de la muestra.
- {interpretar_asimetria(est)}
- Un **atípico leve** cae entre 1.5 y 3 IQR fuera de la caja; un **atípico extremo** supera los 3 IQR (por debajo de {est['li_ext']:.3f} o por encima de {est['ls_ext']:.3f}).
"""
    )
    if not serie_atipicos.empty:
        st.warning(
            "En sistemas reales, estos atípicos pueden indicar fallas de refrigeración, sobrecargas de procesamiento, "
            "fallos de hardware o errores de medición. Conviene investigarlos antes de eliminarlos."
        )