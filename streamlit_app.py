import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eficiência de Recursos — PMEs EU",
    page_icon="🌿",
    layout="wide",
)

# ── Constants (mirror notebook definitions) ───────────────────────────────────
Q1_MAP = {
    'q1.1': 'Poupar água',
    'q1.2': 'Poupar energia',
    'q1.3': 'Energia renovável',
    'q1.4': 'Poupar materiais',
    'q1.5': 'Fornecedores mais verdes',
    'q1.6': 'Minimizar resíduos',
    'q1.7': 'Vender resíduos',
    'q1.8': 'Reciclar internamente',
    'q1.9': 'Eco-design',
}
Q1_COLS   = [f'{k}_bin' for k in Q1_MAP]
Q1_LABELS = list(Q1_MAP.values())

SECTOR_MAP = {1.0: 'Manufatura', 2.0: 'Indústria', 3.0: 'Retalho', 4.0: 'Serviços'}
SIZE_MAP   = {1.0: 'Micro (1–9)', 2.0: 'Pequena (10–49)', 3.0: 'Média (50–249)'}
INVEST_ORDER = ['Nenhum', '<1%', '1–5%', '6–10%', '11–30%', '>30%', 'Ns/Nr']
INVEST_MAP = {0: 'Nenhum', 1: '<1%', 2: '1–5%', 3: '6–10%', 4: '11–30%', 5: '>30%', 6: 'Ns/Nr'}

ISO2_TO_ISO3 = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'CY': 'CYP', 'CZ': 'CZE',
    'DE': 'DEU', 'DK': 'DNK', 'EE': 'EST', 'GR': 'GRC', 'ES': 'ESP',
    'FI': 'FIN', 'FR': 'FRA', 'HR': 'HRV', 'HU': 'HUN', 'IE': 'IRL',
    'IT': 'ITA', 'LT': 'LTU', 'LU': 'LUX', 'LV': 'LVA', 'MT': 'MLT',
    'NL': 'NLD', 'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'SE': 'SWE',
    'SI': 'SVN', 'SK': 'SVK', 'GB': 'GBR',
}
COUNTRY_NAMES = {
    'AT': 'Áustria',      'BE': 'Bélgica',       'BG': 'Bulgária',
    'CY': 'Chipre',       'CZ': 'Chéquia',        'DE': 'Alemanha',
    'DK': 'Dinamarca',    'EE': 'Estónia',         'GR': 'Grécia',
    'ES': 'Espanha',      'FI': 'Finlândia',       'FR': 'França',
    'HR': 'Croácia',      'HU': 'Hungria',          'IE': 'Irlanda',
    'IT': 'Itália',       'LT': 'Lituânia',         'LU': 'Luxemburgo',
    'LV': 'Letónia',      'MT': 'Malta',            'NL': 'Países Baixos',
    'PL': 'Polónia',      'PT': 'Portugal',          'RO': 'Roménia',
    'SE': 'Suécia',       'SI': 'Eslovénia',         'SK': 'Eslováquia',
    'GB': 'Reino Unido',
}

BLUE   = '#4C6EFF'
ORANGE = '#FF8C42'
TMPL   = 'plotly_dark'

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/df_model_app.csv')
    df['iso3']         = df['isocntry'].map(ISO2_TO_ISO3)
    df['País']         = df['isocntry'].map(COUNTRY_NAMES)
    df['Setor']        = df['nace_b'].map(SECTOR_MAP)
    df['Dimensão']     = df['scr10'].map(SIZE_MAP)
    df['Investimento'] = df['investment_code'].map(INVEST_MAP)
    return df

df = load_data()
q1_avail  = [c for c in Q1_COLS   if c in df.columns]
q1_labels = [Q1_MAP[c.replace('_bin', '')] for c in q1_avail]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Filtros")
    sel_sector = st.selectbox("Setor", ['Todos'] + list(SECTOR_MAP.values()))
    sel_size   = st.selectbox("Dimensão", ['Todas'] + list(SIZE_MAP.values()))
    st.divider()

df_f = df.copy()
if sel_sector != 'Todos':
    df_f = df_f[df_f['Setor'] == sel_sector]
if sel_size != 'Todas':
    df_f = df_f[df_f['Dimensão'] == sel_size]

with st.sidebar:
    st.metric("Empresas seleccionadas", f"{len(df_f):,}")
    st.metric("Países", df_f['isocntry'].nunique())

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌿 Eficiência de Recursos em PMEs Europeias")
st.caption("Flash Eurobarometer 549 · Junho 2024 · EU27 + Reino Unido · GESIS ZA8869")
st.divider()

tab_map, tab_country, tab_compare = st.tabs(["🗺️ Mapa Europa", "📊 Por País", "📈 Comparação"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EU MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    col_map, col_ctrl = st.columns([4, 1])
    with col_ctrl:
        st.markdown("**Métrica**")
        map_metric = st.radio(
            "metric", ["MCA Score", "Índice de Intensidade", "Taxa de adopção (%)"],
            label_visibility="collapsed",
        )

    agg = (
        df_f.groupby(['isocntry', 'País', 'iso3'])
        .agg(mca=('mca_score', 'mean'), intensity=('intensity_index', 'mean'), n=('mca_score', 'count'))
        .reset_index()
    )
    adopt_by_country = df_f.groupby('isocntry')[q1_avail].mean().mean(axis=1) * 100
    agg['adoption'] = agg['isocntry'].map(adopt_by_country)

    metric_col = {'MCA Score': 'mca', 'Índice de Intensidade': 'intensity', 'Taxa de adopção (%)': 'adoption'}[map_metric]
    color_label = {'MCA Score': 'MCA Score', 'Índice de Intensidade': 'Intensidade', 'Taxa de adopção (%)': 'Adopção (%)'}[map_metric]

    fig_map = px.choropleth(
        agg,
        locations='iso3',
        color=metric_col,
        hover_name='País',
        hover_data={'iso3': False, 'n': ':,', 'mca': ':.3f', 'intensity': ':.2f', 'adoption': ':.1f'},
        color_continuous_scale='Blues',
        scope='europe',
        template=TMPL,
        labels={metric_col: color_label, 'n': 'N', 'mca': 'MCA', 'intensity': 'Intensidade', 'adoption': 'Adopção (%)'},
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=500,
        geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False,
                 lakecolor='rgba(0,0,0,0)', landcolor='rgba(60,60,60,0.3)'),
    )
    with col_map:
        st.plotly_chart(fig_map, use_container_width=True)

    # Note: mca_score is mean-centered by MCA construction (mean ≈ 0 across all firms).
    # Meaningful summaries are std dev (spread) and country-level deviations from 0.
    mca_std  = df_f['mca_score'].std()
    mca_min  = df_f.groupby('isocntry')['mca_score'].mean().min()
    mca_max  = df_f.groupby('isocntry')['mca_score'].mean().max()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MCA Score — Dispersão (DP)", f"{mca_std:.3f}",
              help="Score centrado em 0 por construção — o desvio padrão mede a dispersão entre empresas.")
    c2.metric("Intensidade média (EU)", f"{df_f['intensity_index'].mean():.2f} / 9")
    c3.metric("% adopção média (EU)", f"{df_f[q1_avail].mean().mean()*100:.1f}%")
    c4.metric("N empresas", f"{len(df_f):,}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BY COUNTRY
# ══════════════════════════════════════════════════════════════════════════════
with tab_country:
    country_opts = sorted(df_f['País'].dropna().unique())
    default_idx  = country_opts.index('Portugal') if 'Portugal' in country_opts else 0
    sel_country  = st.selectbox("País", country_opts, index=default_idx)

    df_c   = df_f[df_f['País'] == sel_country]
    eu_mca = df_f['mca_score'].mean()          # ≈ 0 by MCA construction
    c_mca  = df_c['mca_score'].mean()

    m1, m2, m3, m4 = st.columns(4)
    diff = c_mca - eu_mca
    delta_str = (f"+{diff:.3f} acima da média" if diff >= 0
                 else f"{diff:.3f} abaixo da média")
    m1.metric(
        "Desvio MCA vs média EU",
        f"{c_mca:+.3f}",
        delta=delta_str,
        help="Score MCA centrado em 0 — valores positivos indicam adoção mais intensa/rara que a média EU.",
    )
    m2.metric("Intensidade média", f"{df_c['intensity_index'].mean():.2f} / 9")
    m3.metric("N empresas", f"{len(df_c):,}")
    m4.metric("Taxa de adopção", f"{df_c[q1_avail].mean().mean()*100:.1f}%")

    st.divider()
    cl, cr = st.columns(2)

    # Q1 adoption — country vs EU
    with cl:
        c_rates  = df_c[q1_avail].mean() * 100
        eu_rates = df_f[q1_avail].mean() * 100
        fig_q1 = go.Figure([
            go.Bar(name=sel_country, x=q1_labels, y=c_rates.values,  marker_color=BLUE,   opacity=0.9),
            go.Bar(name='Média EU',  x=q1_labels, y=eu_rates.values, marker_color=ORANGE, opacity=0.75),
        ])
        fig_q1.update_layout(
            barmode='group', template=TMPL,
            title='Adopção por prática Q1 (%)',
            yaxis_title='% PMEs', xaxis_tickangle=-30,
            legend=dict(orientation='h', y=1.12), margin=dict(b=90), height=400,
        )
        st.plotly_chart(fig_q1, use_container_width=True)

    # Sector + size breakdown
    with cr:
        sub1, sub2 = st.tabs(["Setor", "Dimensão"])
        with sub1:
            sec_pct = df_c['Setor'].value_counts(normalize=True).mul(100).reset_index()
            sec_pct.columns = ['Setor', 'Percentagem']
            fig_sec = px.bar(sec_pct, x='Setor', y='Percentagem', template=TMPL,
                             color='Setor', color_discrete_sequence=px.colors.sequential.Blues_r,
                             title=f'Distribuição por Setor', height=340)
            fig_sec.update_layout(showlegend=False, yaxis_title='% PMEs')
            st.plotly_chart(fig_sec, use_container_width=True)
        with sub2:
            sz_pct = df_c['Dimensão'].value_counts(normalize=True).mul(100).reset_index()
            sz_pct.columns = ['Dimensão', 'Percentagem']
            fig_sz = px.bar(sz_pct, x='Dimensão', y='Percentagem', template=TMPL,
                            color='Dimensão', color_discrete_sequence=px.colors.sequential.Blues_r,
                            title=f'Distribuição por Dimensão', height=340)
            fig_sz.update_layout(showlegend=False, yaxis_title='% PMEs')
            st.plotly_chart(fig_sz, use_container_width=True)

    # Investment distribution
    inv_pct = (
        df_c['Investimento'].value_counts(normalize=True)
        .reindex(INVEST_ORDER, fill_value=0).mul(100).reset_index()
    )
    inv_pct.columns = ['Investimento', 'Percentagem']
    fig_inv = px.bar(inv_pct, x='Investimento', y='Percentagem', template=TMPL,
                     color_discrete_sequence=[BLUE],
                     title='Investimento ambiental (% vol. negócios) — Q4')
    fig_inv.update_layout(yaxis_title='% PMEs', height=320)
    st.plotly_chart(fig_inv, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    all_countries = sorted(df_f['País'].dropna().unique())
    defaults = [c for c in ['Portugal', 'Alemanha', 'Polónia', 'Suécia'] if c in all_countries][:3]
    sel_countries = st.multiselect("Países a comparar (máx. 6)", all_countries,
                                   default=defaults, max_selections=6)

    if len(sel_countries) < 2:
        st.info("Selecione pelo menos 2 países.")
    else:
        df_sel = df_f[df_f['País'].isin(sel_countries)]
        colors = px.colors.qualitative.Plotly[:len(sel_countries)]

        cl, cr = st.columns(2)

        # Radar — Q1 practices
        with cl:
            fig_radar = go.Figure()
            for country, color in zip(sel_countries, colors):
                rates = df_sel[df_sel['País'] == country][q1_avail].mean() * 100
                theta = q1_labels + [q1_labels[0]]
                r     = list(rates.values) + [rates.values[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=r, theta=theta, name=country,
                    fill='toself', opacity=0.55, line_color=color,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                template=TMPL, title='Adopção por prática Q1 — radar (%)',
                legend=dict(orientation='h', y=-0.15), height=450,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # MCA Score bar
        with cr:
            agg_comp = (
                df_sel.groupby('País')['mca_score'].mean()
                .reindex(sel_countries).sort_values(ascending=False)
            )
            eu_avg = df_f['mca_score'].mean()
            fig_bar = go.Figure([
                go.Bar(x=agg_comp.index, y=agg_comp.values,
                       marker_color=colors[:len(agg_comp)], opacity=0.9),
            ])
            fig_bar.add_hline(y=eu_avg, line_dash='dot', line_color=ORANGE,
                              annotation_text=f'Média EU: {eu_avg:.3f}',
                              annotation_position='top right')
            fig_bar.update_layout(
                template=TMPL, title='MCA Score médio por país',
                yaxis_title='MCA Score', showlegend=False, height=450,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Intensity table
        st.subheader("Tabela resumo")
        summary = (
            df_sel.groupby('País')
            .agg(
                MCA_Score    = ('mca_score', 'mean'),
                Intensidade  = ('intensity_index', 'mean'),
                Adopção_pct  = (q1_avail[0], lambda x: df_sel.loc[x.index, q1_avail].mean(axis=1).mean() * 100),
                N_empresas   = ('mca_score', 'count'),
            )
            .reindex(sel_countries)
            .round(3)
        )
        summary.columns = ['MCA Score', 'Intensidade', 'Adopção (%)', 'N empresas']
        st.dataframe(summary, use_container_width=True)
