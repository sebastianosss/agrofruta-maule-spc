from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats


APP_DIR = Path(__file__).resolve().parent
EXCEL_PATH = APP_DIR / "Base de datos taller Python- unidad 3.xlsx"

FORMATS = [250, 500, 1000]
NOZZLES = [f"B{i:02d}" for i in range(1, 7)]

# Tabulated constants used in the workshop for n=10.
A3 = 0.975350
B3 = 0.283706
B4 = 1.716294
C4 = 0.972659

# I-MR constants for moving ranges of order 2.
D2 = 1.128
D3 = 0.0
D4 = 3.267


st.set_page_config(
    page_title="AgroFruta Maule | SPC",
    page_icon="🍊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 0.85rem 1rem;
    }
    [data-testid="stTabs"] button {font-weight: 600;}
    div[data-testid="stPlotlyChart"] {margin: 0.25rem 0 1.25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Cargando hojas del taller...")
def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four source sheets without changing their values."""
    workbook = Path(path)
    if not workbook.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {workbook}. "
            "Ubica el Excel en la misma carpeta que app.py."
        )

    return (
        pd.read_excel(workbook, sheet_name="Produccion"),
        pd.read_excel(workbook, sheet_name="Especificaciones"),
        pd.read_excel(workbook, sheet_name="Inspeccion_lotes"),
        pd.read_excel(workbook, sheet_name="Energia"),
    )


def estadisticos_subgrupos(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate subgroup statistics using the sample standard deviation."""
    return (
        df.groupby(
            [
                "formato_ml",
                "periodo_fase",
                "periodo_global",
                "boquilla",
                "id_subgrupo",
            ]
        )["volumen_ml"]
        .agg(n="count", xbar="mean", s=lambda values: np.std(values, ddof=1))
        .reset_index()
    )


def referencias(subgrupos: pd.DataFrame, excluir_periodo_12: bool = False) -> dict:
    """Build Xbar-S reference limits with the workshop's tabulated constants."""
    base = subgrupos
    if excluir_periodo_12:
        base = base[base["periodo_fase"] != 12]

    xbarbar = base["xbar"].mean()
    sbar = base["s"].mean()
    sigma_hat = sbar / C4
    return {
        "xbarbar": xbarbar,
        "sbar": sbar,
        "sigma_hat": sigma_hat,
        "lcl_x": xbarbar - A3 * sbar,
        "ucl_x": xbarbar + A3 * sbar,
        "lcl_s": B3 * sbar,
        "ucl_s": B4 * sbar,
        "subgrupos": len(base),
        "periodos": base["periodo_fase"].nunique(),
    }


def fuera_de_control(df: pd.DataFrame, value: str, lower: float, upper: float) -> pd.Series:
    return (df[value] < lower) | (df[value] > upper)


def add_control_limits(
    fig: go.Figure,
    lower: float,
    center: float,
    upper: float,
    *,
    prefix: str = "",
    colors: tuple[str, str, str] = ("#d94f45", "#23856d", "#d94f45"),
) -> None:
    fig.add_hline(
        y=upper,
        line_dash="dash",
        line_color=colors[2],
    )
    fig.add_hline(
        y=center,
        line_color=colors[1],
    )
    fig.add_hline(
        y=lower,
        line_dash="dash",
        line_color=colors[0],
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=f"{prefix}Límites UCL/LCL ({lower:.2f}, {upper:.2f})",
            line={"color": colors[0], "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=f"{prefix}CL ({center:.2f})",
            line={"color": colors[1]},
            hoverinfo="skip",
        )
    )


def plot_subgroup_chart(
    df: pd.DataFrame,
    *,
    value: str,
    title: str,
    y_title: str,
    references_to_plot: list[tuple[str, dict, str]],
    signal_reference: dict,
    x_value: str = "periodo_fase",
) -> go.Figure:
    """Draw subgroup points, reference limits, and red special-cause signals."""
    fig = go.Figure()
    lower_key, upper_key = ("lcl_x", "ucl_x") if value == "xbar" else ("lcl_s", "ucl_s")
    center_key = "xbarbar" if value == "xbar" else "sbar"

    signal_mask = fuera_de_control(
        df, value, signal_reference[lower_key], signal_reference[upper_key]
    )
    nozzle_colors = ["#3568a8", "#7764a8", "#25918a", "#8a6c45", "#50835a", "#59758b"]
    signals = []
    for nozzle, color in zip(sorted(df["boquilla"].unique()), nozzle_colors):
        nozzle_data = df[df["boquilla"] == nozzle].sort_values(
            ["periodo_global", "id_subgrupo"]
        )
        nozzle_signals = nozzle_data.loc[signal_mask.reindex(nozzle_data.index, fill_value=False)]
        signals.append(nozzle_signals)
        customdata = nozzle_data[["id_subgrupo", "periodo_global", "boquilla"]].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=nozzle_data[x_value],
                y=nozzle_data[value],
                customdata=customdata,
                mode="lines+markers",
                name=nozzle,
                legendgroup="subgroups",
                showlegend=False,
                marker={"color": color, "size": 5, "opacity": 0.82},
                line={"color": color, "width": 1, "shape": "linear"},
                hovertemplate=(
                    "Subgrupo: %{customdata[0]}<br>"
                    "Periodo global: %{customdata[1]}<br>"
                    "Boquilla: %{customdata[2]}<br>"
                    f"{value}: %{{y:.5f}}<extra></extra>"
                ),
            )
        )

    all_signals = pd.concat(signals) if signals else df.iloc[0:0]
    fig.add_trace(
        go.Scatter(
            x=all_signals[x_value],
            y=all_signals[value],
            customdata=all_signals[["id_subgrupo", "periodo_global", "boquilla"]].to_numpy(),
            mode="markers",
            name="Fuera de control",
            marker={"color": "#d62728", "size": 10, "line": {"color": "white", "width": 1}},
            hovertemplate=(
                "Subgrupo: %{customdata[0]}<br>"
                "Periodo global: %{customdata[1]}<br>"
                "Boquilla: %{customdata[2]}<br>"
                f"{value}: %{{y:.5f}}<extra>Señal</extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines+markers",
            name="Subgrupos (por boquilla)",
            line={"color": "#3568a8", "width": 1},
            marker={"color": "#3568a8", "size": 5},
            hoverinfo="skip",
        )
    )

    for label, ref, color in references_to_plot:
        fig.add_hline(
            y=ref[upper_key],
            line_dash="dash",
            line_color=color,
            opacity=0.9,
        )
        fig.add_hline(
            y=ref[center_key],
            line_color=color,
            opacity=0.95,
        )
        fig.add_hline(
            y=ref[lower_key],
            line_dash="dash",
            line_color=color,
            opacity=0.9,
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"{label} · límites",
                line={"color": color, "dash": "dash"},
                hoverinfo="skip",
            )
        )
        if len(references_to_plot) == 1:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=f"{label} · CL",
                    line={"color": color},
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Periodo Fase",
        yaxis_title=y_title,
        hovermode="closest",
        template="plotly_white",
        height=475,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.16,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 9},
            "title": {"text": ""},
        },
        margin={"l": 24, "r": 24, "t": 125, "b": 38},
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "automargin": True,
            "title": {"font": {"size": 12}},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "rgba(120, 130, 145, 0.18)",
            "zeroline": False,
            "automargin": True,
            "title": {"font": {"size": 12}},
        },
    )
    return fig


def count_signals(df: pd.DataFrame, ref: dict) -> tuple[int, int]:
    x_count = int(fuera_de_control(df, "xbar", ref["lcl_x"], ref["ucl_x"]).sum())
    s_count = int(fuera_de_control(df, "s", ref["lcl_s"], ref["ucl_s"]).sum())
    return x_count, s_count


def capability_table(
    specs: pd.DataFrame, refs: dict[int, dict]
) -> pd.DataFrame:
    """Reproduce the workshop capability calculations for 250 mL and 500 mL."""
    records = []
    for fmt in (250, 500):
        spec = specs.loc[specs["formato_ml"] == fmt].iloc[0]
        ref = refs[fmt]
        lie = float(spec["LIE_volumen_ml"])
        lse = float(spec["LSE_volumen_ml"])
        sigma = ref["sigma_hat"]
        mu = ref["xbarbar"]
        cp = (lse - lie) / (6 * sigma)
        cpu = (lse - mu) / (3 * sigma)
        cpl = (mu - lie) / (3 * sigma)
        cpk = min(cpu, cpl)
        criterion = float(spec["criterio_minimo_Cpk"])
        records.append(
            {
                "Formato (mL)": fmt,
                "LIE": lie,
                "LSE": lse,
                "Tolerancia": lse - lie,
                "Media (mu)": round(mu, 3),
                "Sigma": round(sigma, 3),
                "Cp": round(cp, 4),
                "Cr": round(1 / cp, 4),
                "Cpu": round(cpu, 4),
                "Cpl": round(cpl, 4),
                "Cpk": round(cpk, 4),
                "Criterio Cpk": criterion,
                "Cumple": "SI" if cpk >= criterion else "NO",
            }
        )
    return pd.DataFrame(records)


def render_phase_one(
    sg_phase_one: pd.DataFrame, references_by_format: dict[int, dict[str, dict]]
) -> None:
    st.subheader("Fase I · cartas Xbar-S y depuración")
    st.caption(
        "Cada punto representa un subgrupo de 10 observaciones. Las referencias se calculan "
        "por formato, igual que en el cuaderno; el filtro de boquilla solo afecta la vista."
    )

    with st.sidebar:
        st.header("Filtros")
        selected_format = st.selectbox("Formato", FORMATS, index=1, key="phase_one_format")
        selected_nozzle = st.selectbox(
            "Boquilla", ["Todas"] + NOZZLES, key="phase_one_nozzle"
        )
        compare_references = st.checkbox(
            "Comparar referencias preliminares y definitivas",
            value=True,
            help="Al desactivarlo se muestra la referencia definitiva; P12 se excluye del cálculo y de la carta de 500 mL.",
        )

    format_data = sg_phase_one.loc[sg_phase_one["formato_ml"] == selected_format].copy()
    display_data = format_data
    if selected_nozzle != "Todas":
        display_data = display_data.loc[display_data["boquilla"] == selected_nozzle]

    preliminary = references_by_format[selected_format]["preliminar"]
    definitive = references_by_format[selected_format]["definitiva"]
    if compare_references:
        shown_reference = preliminary
        reference_lines = [
            ("Preliminar", preliminary, "#d94f45"),
            ("Definitiva", definitive, "#23856d"),
        ]
    else:
        shown_reference = definitive
        reference_lines = [("Definitiva", definitive, "#23856d")]
        if selected_format == 500:
            display_data = display_data.loc[display_data["periodo_fase"] != 12]

    if compare_references:
        preliminary_signals = count_signals(display_data, preliminary)
        definitive_data = display_data
        if selected_format == 500:
            definitive_data = definitive_data.loc[definitive_data["periodo_fase"] != 12]
        definitive_signals = count_signals(definitive_data, definitive)
        signal_text = (
            f"{sum(preliminary_signals)} → {sum(definitive_signals)}"
            if selected_format == 500
            else f"{sum(preliminary_signals)} → {sum(definitive_signals)}"
        )
        signal_label = "Señales I pre→def"
    else:
        x_signals, s_signals = count_signals(display_data, definitive)
        signal_text = f"{x_signals + s_signals} ({x_signals} Xbar · {s_signals} S)"
        signal_label = "Señales en referencia definitiva"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Xbarbar (mL)", f"{shown_reference['xbarbar']:.3f}")
    c2.metric("Sbar (mL)", f"{shown_reference['sbar']:.3f}")
    c3.metric("sigma_hat (mL)", f"{shown_reference['sigma_hat']:.3f}")
    c4.metric(signal_label, signal_text)

    st.caption(
        "Los límites UCL/LCL se muestran con líneas discontinuas y la línea central con línea continua. "
        "Los puntos rojos indican señales fuera de control."
    )
    st.markdown(f"#### Carta Xbar · Fase I · {selected_format} mL")
    st.plotly_chart(
        plot_subgroup_chart(
            display_data,
            value="xbar",
            title="",
            y_title="Promedio de volumen (mL)",
            references_to_plot=reference_lines,
            signal_reference=preliminary if compare_references else definitive,
        ),
        width="stretch",
    )
    st.markdown(f"#### Carta S · Fase I · {selected_format} mL")
    st.plotly_chart(
        plot_subgroup_chart(
            display_data,
            value="s",
            title="",
            y_title="Desviación estándar muestral (mL)",
            references_to_plot=reference_lines,
            signal_reference=preliminary if compare_references else definitive,
        ),
        width="stretch",
    )

    reference_rows = []
    for fmt in FORMATS:
        for key, label in (("preliminar", "Preliminar"), ("definitiva", "Definitiva")):
            ref = references_by_format[fmt][key]
            reference_rows.append(
                {
                    "Formato (mL)": fmt,
                    "Referencia": label,
                    "Subgrupos": ref["subgrupos"],
                    "Periodos": ref["periodos"],
                    "Xbarbar": round(ref["xbarbar"], 3),
                    "Sbar": round(ref["sbar"], 3),
                    "sigma_hat": round(ref["sigma_hat"], 3),
                    "LCL_Xbar": round(ref["lcl_x"], 3),
                    "UCL_Xbar": round(ref["ucl_x"], 3),
                    "LCL_S": round(ref["lcl_s"], 3),
                    "UCL_S": round(ref["ucl_s"], 3),
                }
            )
    st.markdown("**Tabla de referencias Fase I**")
    st.dataframe(pd.DataFrame(reference_rows), width="stretch", hide_index=True)

    if selected_format == 500:
        excluded = format_data.loc[format_data["periodo_fase"] == 12]
        if selected_nozzle != "Todas":
            excluded = excluded.loc[excluded["boquilla"] == selected_nozzle]
        if not excluded.empty:
            st.warning(
                "El periodo 12 de 500 mL se excluye de la referencia definitiva por causa especial. "
                f"Se identifican {len(excluded)} subgrupos del periodo seleccionado."
            )
            st.dataframe(
                excluded[
                    ["id_subgrupo", "periodo_fase", "periodo_global", "boquilla", "xbar", "s"]
                ].round({"xbar": 5, "s": 5}),
                width="stretch",
                hide_index=True,
            )


def render_phase_two(
    sg_phase_two: pd.DataFrame,
    refs: dict[int, dict],
    specs: pd.DataFrame,
    production: pd.DataFrame,
) -> None:
    st.subheader("Fase II · monitoreo y capacidad")
    c_filter, n_filter = st.columns(2)
    selected_format = c_filter.selectbox(
        "Formato", FORMATS, index=1, key="phase_two_format"
    )
    selected_nozzle = n_filter.selectbox(
        "Boquilla", ["Todas"] + NOZZLES, key="phase_two_nozzle"
    )

    data = sg_phase_two.loc[sg_phase_two["formato_ml"] == selected_format].copy()
    if selected_nozzle != "Todas":
        data = data.loc[data["boquilla"] == selected_nozzle]
    ref = refs[selected_format]
    out_x, out_s = count_signals(data, ref)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Xbarbar Fase I (mL)", f"{ref['xbarbar']:.3f}")
    c2.metric("Sbar Fase I (mL)", f"{ref['sbar']:.3f}")
    c3.metric("sigma_hat Fase I (mL)", f"{ref['sigma_hat']:.3f}")
    c4.metric("Señales Fase II", f"{out_x + out_s}", f"Xbar {out_x} · S {out_s}")

    st.caption(
        "Los límites UCL/LCL se muestran con líneas discontinuas y la línea central con línea continua. "
        "Los puntos rojos indican señales fuera de control."
    )
    st.markdown(f"#### Carta Xbar · Fase II · {selected_format} mL")
    st.plotly_chart(
        plot_subgroup_chart(
            data,
            value="xbar",
            title="",
            y_title="Promedio de volumen (mL)",
            references_to_plot=[("Fase I definitiva", ref, "#23856d")],
            signal_reference=ref,
        ),
        width="stretch",
    )
    st.markdown(f"#### Carta S · Fase II · {selected_format} mL")
    st.plotly_chart(
        plot_subgroup_chart(
            data,
            value="s",
            title="",
            y_title="Desviación estándar muestral (mL)",
            references_to_plot=[("Fase I definitiva", ref, "#23856d")],
            signal_reference=ref,
        ),
        width="stretch",
    )

    signal_rows = []
    for fmt in FORMATS:
        subset = sg_phase_two.loc[sg_phase_two["formato_ml"] == fmt]
        x_count, s_count = count_signals(subset, refs[fmt])
        signal_rows.append(
            {"Formato (mL)": fmt, "Señales Xbar": x_count, "Señales S": s_count}
        )
    st.markdown("**Resumen de señales Fase II por formato**")
    st.dataframe(pd.DataFrame(signal_rows), width="stretch", hide_index=True)

    st.markdown("#### Capacidad del proceso frente a especificaciones")
    capability = capability_table(specs, refs)
    st.dataframe(capability, width="stretch", hide_index=True)
    capability_fig = go.Figure()
    capability_fig.add_bar(
        x=capability["Formato (mL)"].astype(str) + " mL",
        y=capability["Cpk"],
        marker_color=["#23856d" if result == "SI" else "#d94f45" for result in capability["Cumple"]],
        name="Cpk",
        text=capability["Cpk"].map(lambda value: f"{value:.4f}"),
        textposition="outside",
    )
    capability_fig.add_hline(
        y=float(capability["Criterio Cpk"].iloc[0]),
        line_dash="dash",
        line_color="#e89b32",
        annotation_text=f"Criterio mínimo Cpk = {capability['Criterio Cpk'].iloc[0]:.2f}",
        annotation_position="top left",
    )
    capability_fig.update_layout(
        title="Índice Cpk y criterio mínimo",
        xaxis_title="Formato",
        yaxis_title="Cpk",
        template="plotly_white",
        showlegend=False,
    )
    st.plotly_chart(capability_fig, width="stretch")

    selected_spec = specs.loc[specs["formato_ml"] == selected_format]
    raw_phase_two = production.loc[
        (production["fase"] == "II")
        & (production["formato_ml"] == selected_format)
    ]
    if selected_nozzle != "Todas":
        raw_phase_two = raw_phase_two.loc[
            raw_phase_two["boquilla"] == selected_nozzle
        ]
    if not selected_spec.empty:
        spec = selected_spec.iloc[0]
        distribution = go.Figure()
        distribution.add_histogram(
            x=raw_phase_two["volumen_ml"],
            name="Volumen Fase II",
            marker_color="#3568a8",
            opacity=0.78,
        )
        distribution.add_vline(
            x=float(spec["LIE_volumen_ml"]),
            line_dash="dash",
            line_color="#d94f45",
            annotation_text=f"LIE {spec['LIE_volumen_ml']:.1f}",
        )
        distribution.add_vline(
            x=float(spec["LSE_volumen_ml"]),
            line_dash="dash",
            line_color="#d94f45",
            annotation_text=f"LSE {spec['LSE_volumen_ml']:.1f}",
        )
        distribution.add_vline(
            x=ref["xbarbar"],
            line_color="#23856d",
            annotation_text=f"Media de referencia {ref['xbarbar']:.3f}",
        )
        distribution.update_layout(
            title=f"Distribución de llenado y especificaciones · {selected_format} mL",
            xaxis_title="Volumen llenado (mL)",
            yaxis_title="Frecuencia",
            template="plotly_white",
            bargap=0.05,
        )
        st.plotly_chart(distribution, width="stretch")

    with st.expander("Carta estandarizada de corridas cortas · boquilla B01"):
        b01 = sg_phase_two.loc[sg_phase_two["boquilla"] == "B01"].sort_values(
            "periodo_global"
        ).copy()
        b01["ref_mean"] = b01["formato_ml"].map(lambda fmt: refs[int(fmt)]["xbarbar"])
        b01["ref_sigma"] = b01["formato_ml"].map(lambda fmt: refs[int(fmt)]["sigma_hat"])
        b01["z"] = (b01["xbar"] - b01["ref_mean"]) / (b01["ref_sigma"] / np.sqrt(10))
        b01["senal"] = b01["z"].abs() > 3
        fig = go.Figure()
        for signal, label, color in (
            (False, "En límites", "#3568a8"),
            (True, "Fuera de control (|Z| > 3)", "#d62728"),
        ):
            points = b01.loc[b01["senal"] == signal]
            fig.add_trace(
                go.Scatter(
                    x=points["periodo_global"],
                    y=points["z"],
                    mode="markers",
                    name=label,
                    customdata=points[
                        ["id_subgrupo", "periodo_fase", "formato_ml", "boquilla", "xbar"]
                    ].to_numpy(),
                    hovertemplate=(
                        "Subgrupo: %{customdata[0]}<br>Periodo Fase: %{customdata[1]}<br>"
                        "Formato: %{customdata[2]} mL<br>Boquilla: %{customdata[3]}<br>"
                        "Xbar: %{customdata[4]:.5f}<br>Z: %{y:.4f}<extra></extra>"
                    ),
                    marker={"color": color, "size": 7},
                )
            )
        for limit in (-3, 0, 3):
            fig.add_hline(
                y=limit,
                line_dash="dash" if limit else "solid",
                line_color="#23856d" if limit == 0 else "#d94f45",
                annotation_text=f"Z = {limit}",
            )
        fig.update_layout(
            title=f"Carta Z de B01 · {int(b01['senal'].sum())} señales",
            xaxis_title="Periodo global (cronológico)",
            yaxis_title="Estadístico Z",
            template="plotly_white",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            b01.loc[b01["senal"], ["periodo_global", "periodo_fase", "formato_ml", "xbar", "z"]],
            width="stretch",
            hide_index=True,
        )


def render_lot_inspection(
    lots: pd.DataFrame, specs: pd.DataFrame
) -> None:
    st.subheader("Inspección de lotes · plan de aceptación")
    c_accept, c_reject = st.columns(2)
    selected_format = st.selectbox(
        "Formato", ["Todos"] + FORMATS, key="lot_format"
    )

    lot_results = (
        lots.groupby(["id_lote", "formato_ml", "N_lote"])
        .agg(n=("unidad_muestra", "count"), d=("cierre_no_conforme", "sum"))
        .reset_index()
    )
    plan_by_format = specs.set_index("formato_ml")["c_plan_simple"].to_dict()
    lot_results["c"] = lot_results["formato_ml"].map(plan_by_format)
    lot_results["decision"] = np.where(
        lot_results["d"] <= lot_results["c"], "Aceptado", "Rechazado"
    )

    visible_lots = lot_results
    if selected_format != "Todos":
        visible_lots = lot_results.loc[lot_results["formato_ml"] == selected_format]
    accepted = int((visible_lots["decision"] == "Aceptado").sum())
    rejected = int((visible_lots["decision"] == "Rechazado").sum())
    total = len(visible_lots)

    c_accept.metric("Lotes inspeccionados", total)
    c_reject.metric(
        "Aceptados / rechazados",
        f"{accepted} / {rejected}",
        f"{accepted / total:.1%} aceptados" if total else "Sin lotes",
    )

    count_by_format = (
        lot_results.pivot_table(
            index="formato_ml",
            columns="decision",
            values="id_lote",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(FORMATS, fill_value=0)
        .reset_index()
    )
    st.markdown("**Decisión observada por formato (plan simple de la hoja Especificaciones)**")
    st.dataframe(count_by_format, width="stretch", hide_index=True)

    decision_counts = (
        lot_results.groupby(["formato_ml", "decision"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=FORMATS, columns=["Aceptado", "Rechazado"], fill_value=0)
    )
    decision_fig = go.Figure()
    for decision, color in (("Aceptado", "#23856d"), ("Rechazado", "#d94f45")):
        decision_fig.add_bar(
            x=[f"{fmt} mL" for fmt in decision_counts.index],
            y=decision_counts[decision],
            name=decision,
            marker_color=color,
        )
    decision_fig.update_layout(
        barmode="stack",
        title="Lotes aceptados y rechazados por formato",
        xaxis_title="Formato",
        yaxis_title="Número de lotes",
        template="plotly_white",
    )
    st.plotly_chart(decision_fig, width="stretch")

    st.markdown("#### Detalle de lotes")
    st.dataframe(
        visible_lots[["id_lote", "formato_ml", "N_lote", "n", "d", "c", "decision"]],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Riesgo del plan y curvas OC")
    risk_rows = []
    p_values = np.linspace(0.0, 0.08, 300)
    selected_for_curve = FORMATS if selected_format == "Todos" else [selected_format]
    oc_fig = go.Figure()
    for fmt in selected_for_curve:
        spec = specs.loc[specs["formato_ml"] == fmt].iloc[0]
        n_base = int(spec["n_plan_simple"])
        c_base = int(spec["c_plan_simple"])
        n_alt = int(spec["n_plan_alternativo"])
        c_alt = int(spec["c_plan_alternativo"])
        nca = float(spec["NCA_plan"])
        ncl = float(spec["NCL_plan"])
        for name, n_plan, c_plan in (
            ("Base", n_base, c_base),
            ("Alternativo", n_alt, c_alt),
        ):
            pa_nca = stats.binom.cdf(c_plan, n_plan, nca)
            pa_ncl = stats.binom.cdf(c_plan, n_plan, ncl)
            risk_rows.append(
                {
                    "Formato (mL)": fmt,
                    "Plan": name,
                    "n": n_plan,
                    "c": c_plan,
                    "Pa(NCA)": round(pa_nca, 4),
                    "alpha": round(1 - pa_nca, 4),
                    "Pa(NCL)": round(pa_ncl, 4),
                    "beta": round(pa_ncl, 4),
                }
            )
            oc_fig.add_scatter(
                x=p_values,
                y=stats.binom.cdf(c_plan, n_plan, p_values),
                mode="lines",
                name=f"{fmt} mL · {name}",
            )

    oc_fig.add_vline(x=0.004, line_dash="dot", line_color="#87909e", annotation_text="NCA 0.004")
    oc_fig.add_vline(x=0.030, line_dash="dot", line_color="#e89b32", annotation_text="NCL 0.030")
    oc_fig.update_layout(
        title="Curvas OC · probabilidad de aceptación",
        xaxis_title="Proporción no conforme (p)",
        yaxis_title="Probabilidad de aceptación Pa(p)",
        template="plotly_white",
    )
    st.plotly_chart(oc_fig, width="stretch")
    st.dataframe(pd.DataFrame(risk_rows), width="stretch", hide_index=True)

    pa_hyper = stats.hypergeom.cdf(2, 600, 18, 200)
    st.info(
        "Escenario B del cuaderno: lote finito N=600, D=18, n=200, c=2; "
        f"Pa = {pa_hyper:.4f} (hipergeométrica)."
    )


def render_energy(energy: pd.DataFrame) -> None:
    st.subheader("Monitoreo de energía · cartas I-MR de residuales")
    phase_one = energy.loc[energy["fase"] == "I"].sort_values("periodo_energia").copy()
    phase_two = energy.loc[energy["fase"] == "II"].sort_values("periodo_energia").copy()
    if len(phase_one) < 2:
        st.error("La hoja Energia requiere al menos dos observaciones de Fase I.")
        return

    regression = stats.linregress(
        phase_one["velocidad_envases_min"], phase_one["consumo_kwh_h"]
    )
    slope = float(regression.slope)
    intercept = float(regression.intercept)
    r_squared = float(regression.rvalue**2)

    all_energy = pd.concat([phase_one, phase_two]).sort_values("periodo_energia").copy()
    all_energy["pred"] = (
        intercept + slope * all_energy["velocidad_envases_min"]
    )
    all_energy["e"] = all_energy["consumo_kwh_h"] - all_energy["pred"]
    all_energy["mr"] = all_energy["e"].diff().abs()

    phase_one_residuals = all_energy.loc[all_energy["fase"] == "I", "e"]
    phase_one_mr = all_energy.loc[all_energy["fase"] == "I", "mr"]
    center = float(phase_one_residuals.mean())
    mr_bar = float(phase_one_mr.mean())
    sigma_e = mr_bar / D2
    lcl_i, ucl_i = center - 3 * sigma_e, center + 3 * sigma_e
    lcl_mr, ucl_mr = D3 * mr_bar, D4 * mr_bar

    all_energy["senal_i"] = (all_energy["e"] < lcl_i) | (all_energy["e"] > ucl_i)
    all_energy["senal_mr"] = all_energy["mr"] > ucl_mr
    phase_two_all = all_energy.loc[all_energy["fase"] == "II"]
    i_signals = phase_two_all.loc[phase_two_all["senal_i"]]
    mr_signals = phase_two_all.loc[phase_two_all["senal_mr"]]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² regresión Fase I", f"{r_squared:.4f}")
    m2.metric("CL residual (kWh/h)", f"{center:.3f}")
    m3.metric("sigma_e (kWh/h)", f"{sigma_e:.3f}")
    m4.metric("Señales Fase II", f"{len(i_signals) + len(mr_signals)}", f"I {len(i_signals)} · MR {len(mr_signals)}")
    st.caption(
        f"Base histórica: consumo = {intercept:.4f} + {slope:.4f} × velocidad; "
        f"MR_bar = {mr_bar:.3f}; n Fase I = {len(phase_one)}, n Fase II = {len(phase_two)}."
    )

    reg_fig = go.Figure()
    for phase, color, label in (("I", "#3568a8", "Fase I"), ("II", "#d94f45", "Fase II")):
        points = energy.loc[energy["fase"] == phase]
        reg_fig.add_scatter(
            x=points["velocidad_envases_min"],
            y=points["consumo_kwh_h"],
            mode="markers",
            name=label,
            marker={"color": color, "size": 7},
            customdata=points[["periodo_energia", "fase"]].to_numpy(),
            hovertemplate=(
                "Periodo: %{customdata[0]}<br>Fase: %{customdata[1]}<br>"
                "Velocidad: %{x:.3f}<br>Consumo: %{y:.3f}<extra></extra>"
            ),
        )
    x_line = np.linspace(
        phase_one["velocidad_envases_min"].min(),
        phase_one["velocidad_envases_min"].max(),
        100,
    )
    reg_fig.add_scatter(
        x=x_line,
        y=intercept + slope * x_line,
        mode="lines",
        name="Ajuste Fase I",
        line={"color": "#23856d", "width": 2},
    )
    reg_fig.update_layout(
        title="Regresión histórica de consumo vs velocidad",
        xaxis_title="Velocidad (envases/min)",
        yaxis_title="Consumo (kWh/h)",
        template="plotly_white",
    )
    st.plotly_chart(reg_fig, width="stretch")

    st.markdown("#### Carta I · residuales")
    fig_i = go.Figure()
    for phase, color, label in (("I", "#3568a8", "Fase I"), ("II", "#d94f45", "Fase II")):
        points = all_energy.loc[(all_energy["fase"] == phase) & ~all_energy["senal_i"]]
        signals = all_energy.loc[(all_energy["fase"] == phase) & all_energy["senal_i"]]
        fig_i.add_scatter(
            x=points["periodo_energia"],
            y=points["e"],
            mode="lines+markers",
            name=label,
            marker={"color": color, "size": 6},
            customdata=points[["fase", "velocidad_envases_min", "consumo_kwh_h"]].to_numpy(),
            hovertemplate=(
                "Periodo: %{x}<br>Fase: %{customdata[0]}<br>Residual: %{y:.5f}<br>"
                "Velocidad: %{customdata[1]:.3f}<br>Consumo: %{customdata[2]:.3f}<extra></extra>"
            ),
        )
        if not signals.empty:
            fig_i.add_scatter(
                x=signals["periodo_energia"],
                y=signals["e"],
                mode="markers",
                name=f"Señal · {label}",
                marker={"color": "#d62728", "size": 10, "symbol": "x"},
                customdata=signals[["fase", "velocidad_envases_min", "consumo_kwh_h"]].to_numpy(),
                hovertemplate=(
                    "Periodo: %{x}<br>Fase: %{customdata[0]}<br>Residual: %{y:.5f}<br>"
                    "Velocidad: %{customdata[1]:.3f}<br>Consumo: %{customdata[2]:.3f}<extra>Señal</extra>"
                ),
                showlegend=False,
            )
    add_control_limits(fig_i, lcl_i, center, ucl_i)
    fig_i.update_layout(
        title=None,
        xaxis_title="Periodo energía",
        yaxis_title="Residual e (kWh/h)",
        template="plotly_white",
        height=440,
        legend={"orientation": "h", "y": 1.14, "yanchor": "bottom", "font": {"size": 10}},
        margin={"l": 24, "r": 24, "t": 120, "b": 38},
        xaxis={"showgrid": False, "automargin": True},
        yaxis={"showgrid": True, "gridcolor": "rgba(120, 130, 145, 0.18)", "automargin": True},
    )
    st.plotly_chart(fig_i, width="stretch")

    st.markdown("#### Carta MR · residuales")
    fig_mr = go.Figure()
    for phase, color, label in (("I", "#3568a8", "Fase I"), ("II", "#d94f45", "Fase II")):
        points = all_energy.loc[
            (all_energy["fase"] == phase) & ~all_energy["senal_mr"] & all_energy["mr"].notna()
        ]
        signals = all_energy.loc[(all_energy["fase"] == phase) & all_energy["senal_mr"]]
        fig_mr.add_scatter(
            x=points["periodo_energia"],
            y=points["mr"],
            mode="lines+markers",
            name=label,
            marker={"color": color, "size": 6},
            customdata=points[["fase", "velocidad_envases_min", "consumo_kwh_h"]].to_numpy(),
            hovertemplate=(
                "Periodo: %{x}<br>Fase: %{customdata[0]}<br>MR: %{y:.5f}<br>"
                "Velocidad: %{customdata[1]:.3f}<br>Consumo: %{customdata[2]:.3f}<extra></extra>"
            ),
        )
        if not signals.empty:
            fig_mr.add_scatter(
                x=signals["periodo_energia"],
                y=signals["mr"],
                mode="markers",
                name=f"Señal · {label}",
                marker={"color": "#d62728", "size": 10, "symbol": "x"},
                customdata=signals[["fase", "velocidad_envases_min", "consumo_kwh_h"]].to_numpy(),
                hovertemplate=(
                    "Periodo: %{x}<br>Fase: %{customdata[0]}<br>MR: %{y:.5f}<br>"
                    "Velocidad: %{customdata[1]:.3f}<br>Consumo: %{customdata[2]:.3f}<extra>Señal</extra>"
                ),
                showlegend=False,
            )
    add_control_limits(fig_mr, lcl_mr, mr_bar, ucl_mr)
    fig_mr.update_layout(
        title=None,
        xaxis_title="Periodo energía",
        yaxis_title="Rango móvil (kWh/h)",
        template="plotly_white",
        height=440,
        legend={"orientation": "h", "y": 1.14, "yanchor": "bottom", "font": {"size": 10}},
        margin={"l": 24, "r": 24, "t": 120, "b": 38},
        xaxis={"showgrid": False, "automargin": True},
        yaxis={"showgrid": True, "gridcolor": "rgba(120, 130, 145, 0.18)", "automargin": True},
    )
    st.plotly_chart(fig_mr, width="stretch")

    st.markdown("**Señales Fase II · Carta I**")
    st.dataframe(
        i_signals[["periodo_energia", "e", "velocidad_envases_min", "consumo_kwh_h"]].round(4),
        width="stretch",
        hide_index=True,
    )
    st.markdown("**Señales Fase II · Carta MR**")
    st.dataframe(
        mr_signals[["periodo_energia", "mr", "velocidad_envases_min", "consumo_kwh_h"]].round(4),
        width="stretch",
        hide_index=True,
    )


try:
    df_prod, df_specs, df_lots, df_energy = load_data(str(EXCEL_PATH))
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"No fue posible cargar el libro de datos: {exc}")
    st.stop()

st.title("AgroFruta Maule")
st.markdown("### IES-325 · Control Estadístico de Procesos")
st.caption("Panel de monitoreo para el análisis y defensa del caso.")

sg_phase_one = estadisticos_subgrupos(df_prod.loc[df_prod["fase"] == "I"])
sg_phase_two = estadisticos_subgrupos(df_prod.loc[df_prod["fase"] == "II"])

references_by_format: dict[int, dict[str, dict]] = {}
for format_ml in FORMATS:
    format_phase_one = sg_phase_one.loc[sg_phase_one["formato_ml"] == format_ml]
    references_by_format[format_ml] = {
        "preliminar": referencias(format_phase_one),
        "definitiva": referencias(
            format_phase_one, excluir_periodo_12=(format_ml == 500)
        ),
    }
definitive_references = {
    format_ml: references_by_format[format_ml]["definitiva"]
    for format_ml in FORMATS
}

phase_tabs = st.tabs(
    [
        "1 · Cartas Xbar-S Fase I",
        "2 · Fase II y capacidad",
        "3 · Inspección de lotes",
        "4 · Energía I-MR",
    ]
)
with phase_tabs[0]:
    render_phase_one(sg_phase_one, references_by_format)
with phase_tabs[1]:
    render_phase_two(sg_phase_two, definitive_references, df_specs, df_prod)
with phase_tabs[2]:
    render_lot_inspection(df_lots, df_specs)
with phase_tabs[3]:
    render_energy(df_energy)
