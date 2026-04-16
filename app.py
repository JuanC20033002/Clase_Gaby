import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Score Crediticio — Nuevo León",
    page_icon="🏦",
    layout="wide",
)

COLOR_MAP = {"Aprobado": "#22c55e", "Rechazado": "#ef4444"}
MESES_ORDEN = ["Enero", "Febrero", "Marzo"]

@st.cache_data
def load_and_clean_data():
    censo = pd.read_csv("censo_inegi.csv")
    enoe = pd.read_csv("enoe_empleo.csv")
    enigh = pd.read_csv("enigh_finanzas.csv")

    censo_use = censo[[
        "id_persona", "año_nacimiento", "estado", "municipio", "lat", "lon"
    ]].copy()

    enoe_use = enoe[[
        "id_persona", "año", "mes", "estatus_laboral", "tipo_contrato",
        "antiguedad_laboral_anios", "ingresos_mensuales_mxn"
    ]].copy()

    enigh_use = enigh[[
        "id_persona", "año", "saldo_cuenta_mxn"
    ]].copy()

    df = censo_use.merge(enoe_use, on="id_persona", how="inner")
    df = df.merge(enigh_use, on=["id_persona", "año"], how="inner")

    df["edad"] = df["año"] - df["año_nacimiento"]

    df = df[
        (df["estado"] == "Nuevo León") &
        (df["edad"].between(18, 27)) &
        (df["estatus_laboral"].isin(["Empleado", "Emprendedor"]))
    ].copy()

    df["puntos_antiguedad"] = df["antiguedad_laboral_anios"] * 10
    df["puntos_ahorro"] = df["saldo_cuenta_mxn"] / 1000
    df["puntos_estabilidad"] = df["tipo_contrato"].map({
        "Indefinido": 20,
        "Temporal": 5,
        "Por obra": 5,
        "Sin contrato": 0
    }).fillna(0)

    df["Score_Final"] = (
        df["puntos_antiguedad"] +
        df["puntos_ahorro"] +
        df["puntos_estabilidad"]
    ).round(2)

    df["Aprobado"] = (
        (df["Score_Final"] > 70) &
        (df["ingresos_mensuales_mxn"] > 12000)
    )

    df["Estatus_Aprobacion"] = df["Aprobado"].map({
        True: "Aprobado",
        False: "Rechazado"
    })

    df["mes"] = pd.Categorical(df["mes"], categories=MESES_ORDEN, ordered=True)
    df = df.sort_values(["año", "mes", "municipio"]).reset_index(drop=True)
    df["id_solicitud"] = range(1, len(df) + 1)

    return df


df = load_and_clean_data()

st.sidebar.title("Dashboard Crediticio")
st.sidebar.markdown("**Estado:** Nuevo León")
st.sidebar.markdown("**Segmento:** 18-27 años")
st.sidebar.markdown("**Estatus laboral:** Empleado / Emprendedor")
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegar a",
    [
        "📍 Monitor de Aprobación",
        "📊 Análisis de Riesgo",
        "🗓️ Dinámica Temporal",
    ]
)

años_disponibles = sorted(df["año"].unique())
años_sel = st.sidebar.multiselect(
    "Filtrar por año",
    options=años_disponibles,
    default=años_disponibles
)

if not años_sel:
    st.warning("Selecciona al menos un año en la barra lateral.")
    st.stop()

df_filtrado = df[df["año"].isin(años_sel)].copy()

st.sidebar.divider()
st.sidebar.caption(
    "Fuentes base cargadas:\n"
    "- censo_inegi.csv\n"
    "- enoe_empleo.csv\n"
    "- enigh_finanzas.csv"
)

if pagina == "📍 Monitor de Aprobación":
    st.title("📍 Monitor de Aprobación — Nuevo León")
    st.markdown("Datos integrados y limpiados dentro de la app a partir de 3 bases fuente.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Solicitudes", f"{len(df_filtrado):,}")
    c2.metric("Aprobados", f"{int(df_filtrado['Aprobado'].sum()):,}")
    c3.metric("Tasa aprobación", f"{df_filtrado['Aprobado'].mean()*100:.1f}%")
    c4.metric("Ingreso promedio", f"${df_filtrado['ingresos_mensuales_mxn'].mean():,.0f} MXN")

    st.divider()

    fig = px.scatter_map(
        df_filtrado,
        lat="lat",
        lon="lon",
        color="Estatus_Aprobacion",
        color_discrete_map=COLOR_MAP,
        hover_name="municipio",
        hover_data={
            "año": True,
            "mes": True,
            "edad": True,
            "Score_Final": True,
            "ingresos_mensuales_mxn": True,
            "estatus_laboral": True,
            "tipo_contrato": True,
            "lat": False,
            "lon": False,
        },
        zoom=7,
        height=600,
        title="Solicitudes por ubicación geográfica"
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(r=0, t=50, l=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resumen por municipio")
    resumen = (
        df_filtrado.groupby("municipio")
        .agg(
            Solicitudes=("id_solicitud", "count"),
            Aprobados=("Aprobado", "sum"),
            Ingreso_Promedio=("ingresos_mensuales_mxn", "mean"),
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

elif pagina == "📊 Análisis de Riesgo":
    st.title("📊 Análisis de Riesgo")

    ingreso_min = st.slider(
        "Ajusta el ingreso mínimo requerido",
        min_value=7000,
        max_value=30000,
        value=12000,
        step=500,
    )

    df_r = df_filtrado.copy()
    df_r["Aprobado_Dinamico"] = (
        (df_r["Score_Final"] > 70) &
        (df_r["ingresos_mensuales_mxn"] > ingreso_min)
    )
    df_r["Estatus_Dinamico"] = df_r["Aprobado_Dinamico"].map({
        True: "Aprobado",
        False: "Rechazado"
    })

    tasa_dinamica = df_r["Aprobado_Dinamico"].mean() * 100
    st.info(
        f"Con ingreso mínimo de ${ingreso_min:,.0f} MXN, la tasa de aprobación es {tasa_dinamica:.1f}% "
        f"({int(df_r['Aprobado_Dinamico'].sum())}/{len(df_r)} solicitudes)."
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_scatter = px.scatter(
            df_r,
            x="ingresos_mensuales_mxn",
            y="Score_Final",
            color="Estatus_Dinamico",
            color_discrete_map=COLOR_MAP,
            hover_name="municipio",
            size="saldo_cuenta_mxn",
            size_max=18,
            hover_data={
                "año": True,
                "mes": True,
                "edad": True,
                "tipo_contrato": True,
                "antiguedad_laboral_anios": True,
            },
            title="Ingresos vs Score_Final",
            labels={
                "ingresos_mensuales_mxn": "Ingresos mensuales (MXN)",
                "Score_Final": "Score Final",
            },
            height=500,
        )
        fig_scatter.add_vline(
            x=ingreso_min,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Ingreso mínimo: ${ingreso_min:,.0f}",
            annotation_position="top right"
        )
        fig_scatter.add_hline(
            y=70,
            line_dash="dash",
            line_color="blue",
            annotation_text="Score mínimo: 70",
            annotation_position="right"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        fig_map = px.scatter_map(
            df_r,
            lat="lat",
            lon="lon",
            color="Estatus_Dinamico",
            color_discrete_map=COLOR_MAP,
            hover_name="municipio",
            hover_data={
                "año": True,
                "Score_Final": True,
                "ingresos_mensuales_mxn": True,
                "lat": False,
                "lon": False,
            },
            zoom=7,
            height=500,
            title="Impacto del ingreso mínimo en el mapa"
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin=dict(r=0, t=50, l=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Distribución del score por tipo de contrato")
    fig_box = px.box(
        df_filtrado,
        x="tipo_contrato",
        y="Score_Final",
        color="Estatus_Aprobacion",
        color_discrete_map=COLOR_MAP,
        points="all",
        hover_data=["municipio", "año", "ingresos_mensuales_mxn"],
        height=400,
    )
    st.plotly_chart(fig_box, use_container_width=True)

elif pagina == "🗓️ Dinámica Temporal":
    st.title("🗓️ Dinámica Temporal")
    st.markdown("Para la animación, selecciona un solo año.")

    año_anim = st.selectbox("Selecciona el año para animar", sorted(df_filtrado["año"].unique()))
    df_anim = df_filtrado[df_filtrado["año"] == año_anim].copy()
    df_anim = df_anim.sort_values("mes")

    fig_anim = px.scatter_map(
        df_anim,
        lat="lat",
        lon="lon",
        color="Estatus_Aprobacion",
        color_discrete_map=COLOR_MAP,
        animation_frame="mes",
        hover_name="municipio",
        hover_data={
            "año": True,
            "edad": True,
            "Score_Final": True,
            "ingresos_mensuales_mxn": True,
            "tipo_contrato": True,
            "lat": False,
            "lon": False,
        },
        zoom=7,
        height=650,
        title=f"Flujo mensual de solicitudes — {año_anim}"
    )
    fig_anim.update_layout(mapbox_style="open-street-map", margin=dict(r=0, t=50, l=0, b=0))
    st.plotly_chart(fig_anim, use_container_width=True)

    st.subheader("Métricas mensuales")
    resumen_mes = (
        df_anim.groupby("mes", observed=True)
        .agg(
            Solicitudes=("id_solicitud", "count"),
            Aprobados=("Aprobado", "sum"),
            Ingreso_Promedio=("ingresos_mensuales_mxn", "mean"),
            Score_Promedio=("Score_Final", "mean"),
        )
        .assign(
            Tasa_Aprobacion=lambda x: (x["Aprobados"] / x["Solicitudes"] * 100).round(1),
            Ingreso_Promedio=lambda x: x["Ingreso_Promedio"].round(0).astype(int),
            Score_Promedio=lambda x: x["Score_Promedio"].round(1),
        )
        .reset_index()
    )
    st.dataframe(resumen_mes, use_container_width=True, hide_index=True)

    fig_bar = px.bar(
        resumen_mes,
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
st.caption(
    "Proyecto académico · Integración de 3 bases (Censo, ENOE y ENIGH) · "
    "Limpieza y modelado realizados dentro de app.py"
)