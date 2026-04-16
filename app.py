import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(
    page_title="Dashboard Crediticio NL",
    page_icon="📊",
    layout="wide",
)

FILE_BASE = "base_crediticia_nl.csv"   # ajusta aquí si tu archivo tiene otro nombre
COLOR_MAP = {"Aprobado": "#22c55e", "Rechazado": "#ef4444"}
MESES_ORDEN = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
               "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]

# ────────────────────────────────────────────────────────────────
# Carga base ya limpia
# ────────────────────────────────────────────────────────────────
if not os.path.exists(FILE_BASE):
    st.error(f"No se encontró el archivo '{FILE_BASE}' en el repo.")
    st.stop()

@st.cache_data(show_spinner="Cargando base_crediticia_nl.csv...")
def load_data():
    df = pd.read_csv(FILE_BASE)
    # Normalizar tipos
    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce").fillna(2025).astype(int)
    if "mes" in df.columns:
        df["mes"] = df["mes"].astype(str)
        # Ordenar meses si coinciden con MESES_ORDEN
        meses_ok = [m for m in MESES_ORDEN if m in df["mes"].unique()]
        if meses_ok:
            df["mes"] = pd.Categorical(df["mes"], categories=meses_ok, ordered=True)
    return df

df = load_data()

if df.empty:
    st.error("La base base_crediticia_nl.csv está vacía.")
    st.stop()

# ────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────
st.sidebar.title("Dashboard Crediticio")
st.sidebar.markdown("**Estado:** Nuevo León")
st.sidebar.markdown("**Edad:** 18 a 27 años")
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Secciones",
    ["📍 Monitor de Aprobación", "📊 Análisis de Riesgo", "🗓️ Dinámica Temporal"],
)

# Filtro por año
if "anio" in df.columns:
    años = sorted(df["anio"].dropna().unique().tolist())
    años_sel = st.sidebar.multiselect("Año", años, default=años)
    if not años_sel:
        st.warning("Selecciona al menos un año.")
        st.stop()
    df_f = df[df["anio"].isin(años_sel)].copy()
else:
    df_f = df.copy()

if df_f.empty:
    st.error("No hay datos después de aplicar los filtros.")
    st.stop()

# ────────────────────────────────────────────────────────────────
# Página 1 – Monitor de Aprobación
# ────────────────────────────────────────────────────────────────
if pagina == "📍 Monitor de Aprobación":
    st.title("📍 Monitor de Aprobación — Nuevo León")

    aprobados = (df_f["Estatus_Aprobacion"] == "Aprobado").sum()
    tasa = aprobados / len(df_f) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Solicitudes", f"{len(df_f):,}")
    c2.metric("Aprobados", f"{aprobados:,}")
    c3.metric("Tasa de aprobación", f"{tasa:.1f}%")
    c4.metric("Ingreso promedio", f"${df_f['ingreso_mensual_mxn'].mean():,.0f} MXN")

    st.divider()

    fig_map = px.scatter_mapbox(
        df_f,
        lat="lat",
        lon="lon",
        color="Estatus_Aprobacion",
        color_discrete_map=COLOR_MAP,
        hover_name="municipio",
        hover_data={
            "Score_Final": True,
            "ingreso_mensual_mxn": True,
            "estatus_laboral": True,
            "tipo_contrato": True,
            "edad": True,
            "lat": False,
            "lon": False,
        },
        zoom=6.8,
        height=600,
        title="Solicitudes por municipio",
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Resumen por municipio")
    resumen = (
        df_f.groupby("municipio")
        .agg(
            Solicitudes=("id_solicitud", "count"),
            Aprobados=("Estatus_Aprobacion", lambda x: (x == "Aprobado").sum()),
            Ingreso_Promedio=("ingreso_mensual_mxn", "mean"),
            Score_Promedio=("Score_Final", "mean"),
        )
        .assign(
            Tasa_Aprobacion=lambda x: (x["Aprobados"] / x["Solicitudes"] * 100).round(1),
            Ingreso_Promedio=lambda x: x["Ingreso_Promedio"].round(0).astype(int),
            Score_Promedio=lambda x: x["Score_Promedio"].round(1),
        )
        .sort_values("Solicitudes", ascending=False)
        .reset_index()
    )
    st.dataframe(resumen, use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────
# Página 2 – Análisis de Riesgo
# ────────────────────────────────────────────────────────────────
elif pagina == "📊 Análisis de Riesgo":
    st.title("📊 Análisis de Riesgo")

    ingreso_min = st.slider(
        "Ingreso mínimo requerido (MXN)",
        int(df_f["ingreso_mensual_mxn"].min()),
        int(df_f["ingreso_mensual_mxn"].max()),
        12000,
        step=500,
    )

    df_r = df_f.copy()
    df_r["Estatus_Din"] = np.where(
        (df_r["Score_Final"] > 70) & (df_r["ingreso_mensual_mxn"] > ingreso_min),
        "Aprobado",
        "Rechazado",
    )
    aprobados_din = (df_r["Estatus_Din"] == "Aprobado").sum()
    tasa_din = aprobados_din / len(df_r) * 100

    st.info(
        f"Con ingreso mínimo de ${ingreso_min:,.0f} MXN, la tasa de aprobación es "
        f"{tasa_din:.1f}% ({aprobados_din}/{len(df_r)} solicitudes)."
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_sc = px.scatter(
            df_r,
            x="ingreso_mensual_mxn",
            y="Score_Final",
            color="Estatus_Din",
            color_discrete_map=COLOR_MAP,
            hover_name="municipio",
            size="saldo_cuenta_mxn",
            size_max=18,
            hover_data={
                "estatus_laboral": True,
                "tipo_contrato": True,
                "edad": True,
            },
            labels={
                "ingreso_mensual_mxn": "Ingresos mensuales (MXN)",
                "Score_Final": "Score Final",
            },
            title="Ingresos vs Score Final",
        )
        fig_sc.add_vline(
            x=ingreso_min,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Mínimo: ${ingreso_min:,.0f}",
        )
        fig_sc.add_hline(
            y=70,
            line_dash="dash",
            line_color="blue",
            annotation_text="Score mínimo: 70",
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        fig_m2 = px.scatter_mapbox(
            df_r,
            lat="lat",
            lon="lon",
            color="Estatus_Din",
            color_discrete_map=COLOR_MAP,
            hover_name="municipio",
            hover_data={
                "Score_Final": True,
                "ingreso_mensual_mxn": True,
                "lat": False,
                "lon": False,
            },
            zoom=6.8,
            height=500,
            title="Impacto del ingreso mínimo en el mapa",
        )
        fig_m2.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_m2, use_container_width=True)

    st.subheader("Distribución del score por tipo de contrato")
    fig_box = px.box(
        df_f,
        x="tipo_contrato",
        y="Score_Final",
        color="Estatus_Aprobacion",
        color_discrete_map=COLOR_MAP,
        points="all",
        hover_data=["municipio", "ingreso_mensual_mxn"],
        height=400,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# Página 3 – Dinámica Temporal
# ────────────────────────────────────────────────────────────────
elif pagina == "🗓️ Dinámica Temporal":
    st.title("🗓️ Dinámica Temporal")

    if "mes" not in df_f.columns:
        st.error("La base no tiene columna 'mes'.")
        st.stop()

    meses_disponibles = list(df_f["mes"].cat.categories) if hasattr(df_f["mes"], "cat") else sorted(df_f["mes"].unique())
    st.markdown("Animación mes a mes usando la columna **mes** de la base.")

    fig_anim = px.scatter_mapbox(
        df_f.sort_values("mes"),
        lat="lat",
        lon="lon",
        color="Estatus_Aprobacion",
        color_discrete_map=COLOR_MAP,
        animation_frame="mes",
        hover_name="municipio",
        hover_data={
            "Score_Final": True,
            "ingreso_mensual_mxn": True,
            "estatus_laboral": True,
            "lat": False,
            "lon": False,
        },
        zoom=6.8,
        height=650,
        title="Flujo de solicitudes mes a mes",
    )
    fig_anim.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_anim, use_container_width=True)

    st.subheader("Métricas mensuales")
    res_mes = (
        df_f.groupby("mes", observed=True)
        .agg(
            Solicitudes=("id_solicitud", "count"),
            Aprobados=("Estatus_Aprobacion", lambda x: (x == "Aprobado").sum()),
            Ingreso_Promedio=("ingreso_mensual_mxn", "mean"),
            Score_Promedio=("Score_Final", "mean"),
        )
        .assign(
            Tasa_Aprobacion=lambda x: (x["Aprobados"] / x["Solicitudes"] * 100).round(1),
            Ingreso_Promedio=lambda x: x["Ingreso_Promedio"].round(0).astype(int),
            Score_Promedio=lambda x: x["Score_Promedio"].round(1),
        )
        .reset_index()
    )
    st.dataframe(res_mes, use_container_width=True, hide_index=True)

    fig_bar = px.bar(
        res_mes,
        x="mes",
        y="Tasa_Aprobacion",
        color="Tasa_Aprobacion",
        color_continuous_scale=["#ef4444", "#22c55e"],
        text="Tasa_Aprobacion",
        title="Tasa de aprobación mensual (%)",
        height=350,
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.caption("Proyecto académico · Base integrada y limpia: base_crediticia_nl.csv")