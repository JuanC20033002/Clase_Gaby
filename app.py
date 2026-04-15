import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Score Crediticio — Nuevo León",
    page_icon="🏦",
    layout="wide",
)

@st.cache_data
def load_data():
    df = pd.read_csv("solicitudes_nuevo_leon.csv")
    df["fecha_solicitud"] = pd.to_datetime(df["fecha_solicitud"])
    df = df[
        (df["estado"] == "Nuevo León") &
        (df["edad"].between(18, 27)) &
        (df["estatus_laboral"].isin(["Empleado", "Emprendedor"]))
    ].copy()
    df["puntos_antiguedad"]  = df["antiguedad_laboral_anios"] * 10
    df["puntos_ahorro"]      = df["saldo_cuenta_mxn"] / 1000
    df["puntos_estabilidad"] = df["tipo_contrato"].map({"Indefinido": 20, "Temporal": 5}).fillna(0)
    df["Score_Final"]        = (df["puntos_antiguedad"] + df["puntos_ahorro"] + df["puntos_estabilidad"]).round(2)
    df["Aprobado"]           = (df["Score_Final"] > 70) & (df["ingresos_mensuales_mxn"] > 12000)
    df["Estatus_Aprobacion"] = df["Aprobado"].map({True: "Aprobado", False: "Rechazado"})
    return df

df = load_data()
COLOR_MAP = {"Aprobado": "#22c55e", "Rechazado": "#ef4444"}

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Escudo_de_Nuevo_Le%C3%B3n.svg/200px-Escudo_de_Nuevo_Le%C3%B3n.svg.png",
    width=80,
)
st.sidebar.title("Dashboard Crediticio")
st.sidebar.markdown("**Estado:** Nuevo León  \n**Segmento:** 18-27 años  \n**Estatus:** Empleado / Emprendedor")
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegar a",
    options=[
        "📍 Monitor de Aprobación",
        "📊 Análisis de Riesgo",
        "🗓️ Dinámica Temporal",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Fuentes:\n- Censo INEGI 2020\n- ENOE Q2 2025, INEGI\n- ENIGH 2024, INEGI")

# ── PÁGINA 1 ─────────────────────────────────────────────────────────────────
if pagina == "📍 Monitor de Aprobación":
    st.title("📍 Monitor de Aprobación — Nuevo León")
    st.markdown("Visualización espacial de solicitudes · segmento 18-27 años.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total solicitudes", f"{len(df):,}")
    c2.metric("Aprobados", f"{int(df['Aprobado'].sum()):,}")
    c3.metric("Tasa de aprobación", f"{df['Aprobado'].mean()*100:.1f}%")
    c4.metric("Ingreso promedio", f"${df['ingresos_mensuales_mxn'].mean():,.0f} MXN")

    st.divider()

    fig = px.scatter_map(
        df, lat="lat", lon="lon",
        color="Estatus_Aprobacion", color_discrete_map=COLOR_MAP,
        hover_name="municipio",
        hover_data={"Score_Final":True,"ingresos_mensuales_mxn":True,
                    "edad":True,"estatus_laboral":True,"tipo_contrato":True,
                    "lat":False,"lon":False},
        zoom=8, height=580, title="Solicitudes por ubicación geográfica",
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(r=0,t=50,l=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resumen por municipio")
    resumen = (
        df.groupby("municipio")
        .agg(Total=("id_solicitud","count"), Aprobados=("Aprobado","sum"),
             Ingreso_Promedio=("ingresos_mensuales_mxn","mean"),
             Score_Promedio=("Score_Final","mean"))
        .assign(
            Tasa_Aprobacion=lambda x: (x["Aprobados"]/x["Total"]*100).round(1),
            Ingreso_Promedio=lambda x: x["Ingreso_Promedio"].round(0).astype(int),
            Score_Promedio=lambda x: x["Score_Promedio"].round(1),
        )
        .sort_values("Total", ascending=False).reset_index()
    )
    st.dataframe(resumen, use_container_width=True, hide_index=True)

# ── PÁGINA 2 ─────────────────────────────────────────────────────────────────
elif pagina == "📊 Análisis de Riesgo":
    st.title("📊 Análisis de Riesgo Crediticio")

    ingreso_min = st.slider(
        "💡 Ajusta el ingreso mínimo requerido (MXN)",
        min_value=7_000, max_value=25_000, value=12_000, step=500, format="$%d",
    )

    df_r = df.copy()
    df_r["Aprobado_Din"] = (df_r["Score_Final"] > 70) & (df_r["ingresos_mensuales_mxn"] > ingreso_min)
    df_r["Estatus_Din"]  = df_r["Aprobado_Din"].map({True: "Aprobado", False: "Rechazado"})

    tasa_din = df_r["Aprobado_Din"].mean() * 100
    st.info(f"Con ingreso mínimo **${ingreso_min:,} MXN** → tasa de aprobación: **{tasa_din:.1f}%** ({int(df_r['Aprobado_Din'].sum())}/{len(df_r)})")

    col1, col2 = st.columns(2)
    with col1:
        fig_s = px.scatter(
            df_r, x="ingresos_mensuales_mxn", y="Score_Final",
            color="Estatus_Din", color_discrete_map=COLOR_MAP,
            hover_name="municipio", size="saldo_cuenta_mxn", size_max=18,
            hover_data={"edad":True,"tipo_contrato":True,"antiguedad_laboral_anios":True},
            title="Ingresos vs Score_Final",
            labels={"ingresos_mensuales_mxn":"Ingresos (MXN)","Score_Final":"Score"},
            height=480,
        )
        fig_s.add_vline(x=ingreso_min, line_dash="dash", line_color="orange",
                        annotation_text=f"Ing. mín: ${ingreso_min:,}", annotation_position="top right")
        fig_s.add_hline(y=70, line_dash="dash", line_color="blue",
                        annotation_text="Score mín: 70", annotation_position="right")
        st.plotly_chart(fig_s, use_container_width=True)

    with col2:
        fig_m = px.scatter_map(
            df_r, lat="lat", lon="lon",
            color="Estatus_Din", color_discrete_map=COLOR_MAP,
            hover_name="municipio",
            hover_data={"Score_Final":True,"ingresos_mensuales_mxn":True,"lat":False,"lon":False},
            zoom=7, height=480, title="Impacto del ingreso mínimo en el mapa",
        )
        fig_m.update_layout(mapbox_style="open-street-map", margin=dict(r=0,t=50,l=0,b=0))
        st.plotly_chart(fig_m, use_container_width=True)

    st.subheader("Distribución del Score por tipo de contrato")
    fig_box = px.box(
        df, x="tipo_contrato", y="Score_Final",
        color="Estatus_Aprobacion", color_discrete_map=COLOR_MAP,
        points="all", hover_data=["municipio","ingresos_mensuales_mxn"],
        labels={"tipo_contrato":"Tipo de contrato","Score_Final":"Score Final"},
        height=380,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ── PÁGINA 3 ─────────────────────────────────────────────────────────────────
elif pagina == "🗓️ Dinámica Temporal":
    st.title("🗓️ Dinámica Temporal — Enero a Marzo 2026")

    orden_meses = ["Enero", "Febrero", "Marzo"]
    df["mes"] = pd.Categorical(df["mes"], categories=orden_meses, ordered=True)
    df_sorted  = df.sort_values("mes")

    fig_anim = px.scatter_map(
        df_sorted, lat="lat", lon="lon",
        color="Estatus_Aprobacion", color_discrete_map=COLOR_MAP,
        animation_frame="mes", hover_name="municipio",
        hover_data={"Score_Final":True,"ingresos_mensuales_mxn":True,
                    "edad":True,"tipo_contrato":True,"lat":False,"lon":False},
        zoom=7, height=620, title="Flujo de solicitudes por mes",
    )
    fig_anim.update_layout(mapbox_style="open-street-map", margin=dict(r=0,t=50,l=0,b=0))
    st.plotly_chart(fig_anim, use_container_width=True)

    st.subheader("Métricas mensuales")
    resumen_mes = (
        df.groupby("mes", observed=True)
        .agg(Solicitudes=("id_solicitud","count"), Aprobados=("Aprobado","sum"),
             Ingreso_Promedio=("ingresos_mensuales_mxn","mean"),
             Score_Promedio=("Score_Final","mean"))
        .assign(
            Tasa_Aprobacion=lambda x: (x["Aprobados"]/x["Solicitudes"]*100).round(1),
            Ingreso_Promedio=lambda x: x["Ingreso_Promedio"].round(0).astype(int),
            Score_Promedio=lambda x: x["Score_Promedio"].round(1),
        ).reset_index()
    )
    st.dataframe(resumen_mes, use_container_width=True, hide_index=True)

    fig_bar = px.bar(
        resumen_mes, x="mes", y="Tasa_Aprobacion",
        color="Tasa_Aprobacion", color_continuous_scale=["#ef4444","#22c55e"],
        text="Tasa_Aprobacion", title="Tasa de aprobación mensual (%)",
        labels={"mes":"Mes","Tasa_Aprobacion":"Tasa (%)"}, height=340,
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.caption("Proyecto académico · Score crediticio Nuevo León · Datos sintéticos con base en INEGI 2025")