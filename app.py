import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Dashboard Crediticio NL", page_icon="📊", layout="wide")

COLOR_MAP = {"Aprobado": "green", "Rechazado": "red"}

@st.cache_data(show_spinner=False)
def load_data():
    censo = pd.read_csv("conjunto_de_datos_iter_19CSV20.csv", encoding="utf-8-sig", low_memory=False)
    enigh = pd.read_csv("conjunto_de_datos_ingresos_enigh2024_ns.csv", low_memory=False)
    enoe = pd.read_csv("conjunto_de_datos_coe1_enoe_2025_4t.csv", low_memory=False)
    return censo, enigh, enoe


def clean_censo(censo):
    c = censo.copy()
    c = c[c["ENTIDAD"].astype(str).str.zfill(2) == "19"].copy()
    c = c[c["POBTOT"].astype(str) != "*"].copy()
    c["POBTOT"] = pd.to_numeric(c["POBTOT"], errors="coerce")
    c = c.dropna(subset=["POBTOT", "LATITUD", "LONGITUD", "MUN", "NOM_MUN"]).copy()
    c = c[c["POBTOT"] > 0].copy()

    c["MUN"] = c["MUN"].astype(str).str.zfill(3)

    lat_parts = c["LATITUD"].astype(str).str.extract(r"(\d+)[°](\d+)[']([\d\.]+)")
    lon_parts = c["LONGITUD"].astype(str).str.extract(r"(\d+)[°](\d+)[']([\d\.]+)")

    c["lat"] = (
        lat_parts[0].astype(float)
        + lat_parts[1].astype(float) / 60
        + lat_parts[2].astype(float) / 3600
    )

    c["lon"] = -(
        lon_parts[0].astype(float)
        + lon_parts[1].astype(float) / 60
        + lon_parts[2].astype(float) / 3600
    )

    muni = (
        c.groupby(["MUN", "NOM_MUN"], as_index=False)
        .agg(
            POBTOT=("POBTOT", "sum"),
            lat=("lat", "mean"),
            lon=("lon", "mean")
        )
    )

    muni["estado"] = "Nuevo León"
    muni = muni.rename(columns={"MUN": "cve_mun", "NOM_MUN": "municipio"})
    return muni


def clean_enigh(enigh):
    e = enigh.copy()
    e["entidad"] = e["entidad"].astype(str).str.zfill(2)
    e = e[e["entidad"] == "19"].copy()
    e["numren"] = e["numren"].astype(str).str.zfill(2)

    for col in ["ing_1", "ing_2", "ing_3", "ing_4", "ing_5", "ing_6", "ing_tri"]:
        e[col] = pd.to_numeric(e[col], errors="coerce")

    e["ingreso_mensual_estimado"] = e[
        ["ing_1", "ing_2", "ing_3", "ing_4", "ing_5", "ing_6"]
    ].mean(axis=1, skipna=True)

    e["ingreso_mensual_estimado"] = e["ingreso_mensual_estimado"].fillna(e["ing_tri"] / 3)

    ingreso_persona = (
        e.groupby(["folioviv", "foliohog", "numren"], as_index=False)
        .agg(
            ingreso_mensual_mxn=("ingreso_mensual_estimado", "sum"),
            ingreso_trimestral_mxn=("ing_tri", "sum")
        )
    )

    return ingreso_persona


def clean_enoe(enoe):
    d = enoe.copy()
    d["cve_ent"] = d["cve_ent"].astype(str).str.zfill(2)
    d = d[d["cve_ent"] == "19"].copy()
    d["cve_mun"] = d["cve_mun"].astype(str).str.zfill(3)
    d["n_ren"] = d["n_ren"].astype(str).str.zfill(2)
    d["eda"] = pd.to_numeric(d["eda"], errors="coerce")
    d = d[d["eda"].between(18, 27)].copy()

    d["anio"] = 2025
    d["trimestre"] = "4T"

    d["p3"] = pd.to_numeric(d["p3"], errors="coerce")
    d["p4a"] = pd.to_numeric(d["p4a"], errors="coerce")
    d["p2k_anio"] = pd.to_numeric(d["p2k_anio"], errors="coerce")
    d["p2k_mes"] = pd.to_numeric(d["p2k_mes"], errors="coerce")
    d["p5b_thrs"] = pd.to_numeric(d["p5b_thrs"], errors="coerce")

    d["estatus_laboral"] = np.select(
        [d["p3"] == 1, d["p3"] == 2],
        ["Empleado", "Emprendedor"],
        default="Otro"
    )

    d["tipo_contrato"] = np.select(
        [d["p4a"] == 1, d["p4a"] == 2],
        ["Indefinido", "Temporal"],
        default="Sin contrato"
    )

    d["antiguedad_laboral_anios"] = d["p2k_anio"].fillna(0) + (d["p2k_mes"].fillna(0) / 12)
    d["antiguedad_laboral_anios"] = d["antiguedad_laboral_anios"].clip(lower=0)

    keep = [
        "cve_mun", "n_ren", "eda", "anio", "trimestre",
        "estatus_laboral", "tipo_contrato", "antiguedad_laboral_anios", "p5b_thrs"
    ]
    return d[keep].copy()


def build_final_dataset(censo, enigh, enoe):
    geo = clean_censo(censo)
    inc = clean_enigh(enigh)
    labor = clean_enoe(enoe)

    inc = inc.reset_index(drop=True)
    labor = labor.reset_index(drop=True)

    n = min(len(inc), len(labor))
    inc = inc.iloc[:n].copy()
    labor = labor.iloc[:n].copy()

    df = pd.concat([labor.reset_index(drop=True), inc.reset_index(drop=True)], axis=1)
    df = df.merge(geo, on="cve_mun", how="left")

    df = df[df["estatus_laboral"].isin(["Empleado", "Emprendedor"])].copy()
    df = df.dropna(subset=["lat", "lon", "ingreso_mensual_mxn"]).copy()

    rng = np.random.default_rng(42)
    df["saldo_cuenta_mxn"] = np.where(
        df["ingreso_mensual_mxn"] > 0,
        (df["ingreso_mensual_mxn"] * rng.uniform(0.5, 3.5, len(df))).round(0),
        0
    )

    df["Score_Final"] = (
        df["antiguedad_laboral_anios"] * 10
        + (df["saldo_cuenta_mxn"] / 1000)
        + np.select(
            [df["tipo_contrato"] == "Indefinido", df["tipo_contrato"] == "Temporal"],
            [20, 5],
            default=0
        )
    ).round(2)

    df["Estatus_Aprobacion"] = np.where(
        (df["Score_Final"] > 70) & (df["ingreso_mensual_mxn"] > 12000),
        "Aprobado",
        "Rechazado"
    )

    df["mes"] = pd.Series(
        rng.choice(["Octubre", "Noviembre", "Diciembre"], len(df), p=[0.34, 0.33, 0.33])
    )
    df["id_solicitud"] = range(1, len(df) + 1)
    df["estado"] = "Nuevo León"

    return df


censo_raw, enigh_raw, enoe_raw = load_data()
df = build_final_dataset(censo_raw, enigh_raw, enoe_raw)

st.sidebar.title("Dashboard crediticio")
st.sidebar.markdown("**Estado:** Nuevo León")
st.sidebar.markdown("**Edad:** 18 a 27 años")
st.sidebar.markdown("**Fuente:** Censo 2020 + ENIGH 2024 + ENOE 2025 4T")

pagina = st.sidebar.radio(
    "Secciones",
    ["Monitor de Aprobación", "Análisis de Riesgo", "Dinámica Temporal"]
)

anios = sorted(df["anio"].dropna().unique().tolist())
anio_sel = st.sidebar.multiselect("Año", anios, default=anios)

if not anio_sel:
    st.warning("Selecciona al menos un año.")
    st.stop()

df_f = df[df["anio"].isin(anio_sel)].copy()

if pagina == "Monitor de Aprobación":
    st.title("Monitor de Aprobación")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Solicitudes", f"{len(df_f):,}")
    c2.metric("Aprobados", f"{(df_f['Estatus_Aprobacion'] == 'Aprobado').sum():,}")
    c3.metric("Tasa de aprobación", f"{((df_f['Estatus_Aprobacion'] == 'Aprobado').mean()*100):.1f}%")
    c4.metric("Ingreso promedio", f"${df_f['ingreso_mensual_mxn'].mean():,.0f} MXN")

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
            "lat": False,
            "lon": False
        },
        zoom=6.8,
        height=600
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

elif pagina == "Análisis de Riesgo":
    st.title("Análisis de Riesgo")

    ingreso_min = st.slider("Ingreso mínimo requerido", 5000, 50000, 12000, 500)

    df_r = df_f.copy()
    df_r["estatus_dinamico"] = np.where(
        (df_r["Score_Final"] > 70) & (df_r["ingreso_mensual_mxn"] > ingreso_min),
        "Aprobado",
        "Rechazado"
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_sc = px.scatter(
            df_r,
            x="ingreso_mensual_mxn",
            y="Score_Final",
            color="estatus_dinamico",
            color_discrete_map=COLOR_MAP,
            hover_name="municipio",
            size="saldo_cuenta_mxn",
            title="Ingresos vs Score"
        )
        fig_sc.add_vline(x=ingreso_min, line_dash="dash", line_color="orange")
        fig_sc.add_hline(y=70, line_dash="dash", line_color="blue")
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        fig_map2 = px.scatter_mapbox(
            df_r,
            lat="lat",
            lon="lon",
            color="estatus_dinamico",
            color_discrete_map=COLOR_MAP,
            hover_name="municipio",
            hover_data={
                "Score_Final": True,
                "ingreso_mensual_mxn": True,
                "lat": False,
                "lon": False
            },
            zoom=6.8,
            height=500
        )
        fig_map2.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_map2, use_container_width=True)

elif pagina == "Dinámica Temporal":
    st.title("Dinámica Temporal")

    orden = ["Octubre", "Noviembre", "Diciembre"]
    df_f["mes"] = pd.Categorical(df_f["mes"], categories=orden, ordered=True)

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
            "lat": False,
            "lon": False
        },
        zoom=6.8,
        height=650
    )
    fig_anim.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_anim, use_container_width=True)

    resumen = df_f.groupby("mes", observed=True).agg(
        solicitudes=("id_solicitud", "count"),
        aprobados=("Estatus_Aprobacion", lambda x: (x == "Aprobado").sum()),
        ingreso_promedio=("ingreso_mensual_mxn", "mean")
    ).reset_index()

    resumen["tasa_aprobacion"] = (
        resumen["aprobados"] / resumen["solicitudes"] * 100
    ).round(1)

    st.dataframe(resumen, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Proyecto con datos reales cargados desde CSV oficiales de INEGI. "
    "Para este prototipo, el score usa ENOE para perfil laboral, ENIGH para ingresos y Censo para geografía municipal."
)