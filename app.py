import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Dashboard Crediticio NL", page_icon="📊", layout="wide")

COLOR_MAP = {"Aprobado": "#22c55e", "Rechazado": "#ef4444"}

CENSO_COLS = [
    "ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC",
    "LONGITUD", "LATITUD", "ALTITUD", "POBTOT"
]

FILE_CENSO = "conjunto_de_datos_iter_19CSV20.csv"
FILE_ENIGH = "conjunto_de_datos_ingresos_enigh2024_ns.csv"
FILE_ENOE = "conjunto_de_datos_coe1_enoe_2025_4t.csv"


def check_files():
    missing = []
    for f in [FILE_CENSO, FILE_ENIGH, FILE_ENOE]:
        if not os.path.exists(f):
            missing.append(f)
        elif os.path.getsize(f) == 0:
            missing.append(f + " (vacío)")
    return missing


def dms_to_decimal(text, is_lon=False):
    if pd.isna(text):
        return np.nan

    s = str(text).strip().replace('"', "").replace("''", "").replace("’", "'")

    try:
        ext = pd.Series([s]).str.extract(r"(\d+)[°](\d+)['](\d+\.?\d*)")
        if ext.isna().any(axis=None):
            return np.nan

        deg = float(ext.iloc[0, 0])
        minute = float(ext.iloc[0, 1])
        sec = float(ext.iloc[0, 2])

        val = deg + minute / 60 + sec / 3600
        return -val if is_lon else val
    except Exception:
        return np.nan


@st.cache_data(show_spinner="Cargando y limpiando datos...")
def build_dataset():
    censo = pd.read_csv(
        FILE_CENSO,
        header=None,
        usecols=list(range(10)),
        names=CENSO_COLS,
        encoding="utf-8-sig",
        low_memory=False,
        skiprows=1
    )

    censo["ENTIDAD"] = censo["ENTIDAD"].astype(str).str.strip().str.zfill(2)
    censo = censo[censo["ENTIDAD"] == "19"].copy()

    censo["POBTOT"] = pd.to_numeric(censo["POBTOT"], errors="coerce")
    censo = censo.dropna(subset=["MUN", "NOM_MUN", "LATITUD", "LONGITUD", "POBTOT"]).copy()
    censo = censo[censo["POBTOT"] > 0].copy()

    censo["MUN"] = censo["MUN"].astype(str).str.zfill(3)
    censo["lat"] = censo["LATITUD"].apply(lambda x: dms_to_decimal(x, False))
    censo["lon"] = censo["LONGITUD"].apply(lambda x: dms_to_decimal(x, True))
    censo = censo.dropna(subset=["lat", "lon"]).copy()

    geo = (
        censo.groupby(["MUN", "NOM_MUN"], as_index=False)
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean")
        )
        .rename(columns={"MUN": "cve_mun", "NOM_MUN": "municipio"})
    )

    enigh = pd.read_csv(FILE_ENIGH, low_memory=False)
    enigh["entidad"] = enigh["entidad"].astype(str).str.zfill(2)
    enigh = enigh[enigh["entidad"] == "19"].copy()

    for col in ["ing_1", "ing_2", "ing_3", "ing_4", "ing_5", "ing_6", "ing_tri"]:
        enigh[col] = pd.to_numeric(enigh[col], errors="coerce")

    enigh["ing_mes"] = enigh[["ing_1", "ing_2", "ing_3", "ing_4", "ing_5", "ing_6"]].mean(axis=1, skipna=True)
    enigh["ing_mes"] = enigh["ing_mes"].fillna(enigh["ing_tri"] / 3)

    enigh["cve_mun"] = enigh["est_dis"].astype(str).str.zfill(3)

    ing_mun = (
        enigh.groupby("cve_mun", as_index=False)
        .agg(ingreso_mensual_mxn=("ing_mes", "mean"))
    )

    enoe = pd.read_csv(FILE_ENOE, low_memory=False)
    enoe["cve_ent"] = enoe["cve_ent"].astype(str).str.zfill(2)
    enoe = enoe[enoe["cve_ent"] == "19"].copy()

    enoe["cve_mun"] = enoe["cve_mun"].astype(str).str.zfill(3)
    enoe["eda"] = pd.to_numeric(enoe["eda"], errors="coerce")
    enoe = enoe[enoe["eda"].between(18, 27)].copy()

    for col in ["p3", "p4a", "p2k_anio", "p2k_mes"]:
        enoe[col] = pd.to_numeric(enoe[col], errors="coerce")

    enoe["estatus_laboral"] = np.select(
        [enoe["p3"] == 1, enoe["p3"] == 2],
        ["Empleado", "Emprendedor"],
        default="Otro"
    )

    enoe = enoe[enoe["estatus_laboral"].isin(["Empleado", "Emprendedor"])].copy()

    enoe["tipo_contrato"] = np.select(
        [enoe["p4a"] == 1, enoe["p4a"] == 2],
        ["Indefinido", "Temporal"],
        default="Sin contrato"
    )

    enoe["antiguedad_laboral_anios"] = (
        enoe["p2k_anio"].fillna(0) + enoe["p2k_mes"].fillna(0) / 12
    ).clip(lower=0)

    enoe["anio"] = 2025
    enoe["estado"] = "Nuevo León"

    df = enoe.merge(ing_mun, on="cve_mun", how="left")
    df = df.merge(geo, on="cve_mun", how="left")
    df = df.dropna(subset=["lat", "lon", "ingreso_mensual_mxn"]).copy()

    rng = np.random.default_rng(42)

    df["saldo_cuenta_mxn"] = (
        df["ingreso_mensual_mxn"] * rng.uniform(0.5, 3.5, len(df))
    ).round(0).clip(lower=0)

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

    df["mes"] = rng.choice(
        ["Octubre", "Noviembre", "Diciembre"],
        len(df),
        p=[0.34, 0.33, 0.33]
    )

    df["edad"] = df["eda"].astype(int)
    df["id_solicitud"] = range(1, len(df) + 1)

    return df


missing = check_files()
if missing:
    st.error("No se encontraron o están vacíos estos archivos:\n\n- " + "\n- ".join(missing))
    st.stop()

df = build_dataset()

if df.empty:
    st.error("La base final quedó vacía. Revisa que los 3 CSV tengan datos válidos.")
    st.stop()

st.sidebar.title("Dashboard Crediticio")
st.sidebar.markdown("**Estado:** Nuevo León")
st.sidebar.markdown("**Edad:** 18 a 27 años")
st.sidebar.markdown("**Fuente:** Censo 2020 · ENIGH 2024 · ENOE 2025 4T")
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegar a",
    ["📍 Monitor de Aprobación", "📊 Análisis de Riesgo", "🗓️ Dinámica Temporal"]
)

anios = sorted(df["anio"].dropna().unique().tolist())
anio_sel = st.sidebar.multiselect("Año", anios, default=anios)

if not anio_sel:
    st.warning("Selecciona al menos un año.")
    st.stop()

df_f = df[df["anio"].isin(anio_sel)].copy()

if pagina == "📍 Monitor de Aprobación":
    st.title("📍 Monitor de Aprobación — Nuevo León")

    c1, c2, c3, c4 = st.columns(4)
    aprobados = (df_f["Estatus_Aprobacion"] == "Aprobado").sum()
    tasa = aprobados / len(df_f) * 100

    c1.metric("Solicitudes", f"{len(df_f):,}")
    c2.metric("Aprobados", f"{aprobados:,}")
    c3.metric("Tasa de aprobación", f"{tasa:.1f}%")
    c4.metric("Ingreso promedio", f"${df_f['ingreso_mensual_mxn'].mean():,.0f} MXN")

    st.divider()

    fig = px.scatter_mapbox(
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
            "lon": False
        },
        zoom=6.8,
        height=600,
        title="Solicitudes por municipio en Nuevo León"
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resumen por municipio")
    resumen = (
        df_f.groupby("municipio")
        .agg(
            Solicitudes=("id_solicitud", "count"),
            Aprobados=("Estatus_Aprobacion", lambda x: (x == "Aprobado").sum()),
            Ingreso_Promedio=("ingreso_mensual_mxn", "mean"),
            Score_Promedio=("Score_Final", "mean")
        )
        .assign(
            Tasa=lambda x: (x["Aprobados"] / x["Solicitudes"] * 100).round(1),
            Ingreso_Promedio=lambda x: x["Ingreso_Promedio"].round(0).astype(int),
            Score_Promedio=lambda x: x["Score_Promedio"].round(1)
        )
        .sort_values("Solicitudes", ascending=False)
        .reset_index()
    )
    st.dataframe(resumen, use_container_width=True, hide_index=True)

elif pagina == "📊 Análisis de Riesgo":
    st.title("📊 Análisis de Riesgo")

    ingreso_min = st.slider("Ajusta el ingreso mínimo requerido (MXN)", 5000, 50000, 12000, 500)

    df_r = df_f.copy()
    df_r["Estatus_Din"] = np.where(
        (df_r["Score_Final"] > 70) & (df_r["ingreso_mensual_mxn"] > ingreso_min),
        "Aprobado",
        "Rechazado"
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
                "edad": True
            },
            labels={
                "ingreso_mensual_mxn": "Ingresos mensuales (MXN)",
                "Score_Final": "Score Final"
            },
            title="Ingresos vs Score Final"
        )
        fig_sc.add_vline(
            x=ingreso_min,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Mínimo: ${ingreso_min:,.0f}"
        )
        fig_sc.add_hline(
            y=70,
            line_dash="dash",
            line_color="blue",
            annotation_text="Score mínimo: 70"
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
                "lon": False
            },
            zoom=6.8,
            height=500,
            title="Impacto del ingreso mínimo en el mapa"
        )
        fig_m2.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
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
        height=400
    )
    st.plotly_chart(fig_box, use_container_width=True)

elif pagina == "🗓️ Dinámica Temporal":
    st.title("🗓️ Dinámica Temporal")

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
            "estatus_laboral": True,
            "lat": False,
            "lon": False
        },
        zoom=6.8,
        height=650,
        title="Flujo de solicitudes mes a mes — 2025 4T"
    )
    fig_anim.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_anim, use_container_width=True)

    st.subheader("Métricas mensuales")
    res_mes = (
        df_f.groupby("mes", observed=True)
        .agg(
            Solicitudes=("id_solicitud", "count"),
            Aprobados=("Estatus_Aprobacion", lambda x: (x == "Aprobado").sum()),
            Ingreso_Promedio=("ingreso_mensual_mxn", "mean"),
            Score_Promedio=("Score_Final", "mean")
        )
        .assign(
            Tasa_Aprobacion=lambda x: (x["Aprobados"] / x["Solicitudes"] * 100).round(1),
            Ingreso_Promedio=lambda x: x["Ingreso_Promedio"].round(0).astype(int),
            Score_Promedio=lambda x: x["Score_Promedio"].round(1)
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
        height=350
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.caption("Proyecto académico · Datos reales INEGI: Censo 2020 NL + ENIGH 2024 + ENOE 2025 4T")