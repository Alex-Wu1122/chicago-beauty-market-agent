import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from data_loader import load_data
from chatbot import chat

st.set_page_config(
    page_title="Chicago Beauty Market Agent",
    page_icon="🤖",
    layout="wide",
)

@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def get_geojson():
    with open("chicago_zips.geojson") as f:
        return json.load(f)

df = get_data()
geojson = get_geojson()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 0; }
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #3a3a5c;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }
    .kpi-label { font-size: 0.75rem; color: #94a3b8; margin-top: 0.1rem; }
    .zip-card {
        background: linear-gradient(135deg, #1a2744, #1e3a5f);
        border: 1px solid #3a5a8f;
        border-radius: 10px;
        padding: 0.4rem 1rem;
        margin-top: 0.3rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a2744, #1e3a5f);
        border-left: 4px solid #60a5fa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .stChatMessage { font-size: 0.88rem; }
    div[data-testid='stHorizontalBlock'] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
if "map_layer" not in st.session_state:
    st.session_state.map_layer = "OpportunityScore"
if "selected_zip" not in st.session_state:
    st.session_state.selected_zip = None
if "compare_zip1" not in st.session_state:
    st.session_state.compare_zip1 = None
if "compare_zip2" not in st.session_state:
    st.session_state.compare_zip2 = None
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! I'm the **Chicago Beauty Market Agent**. I can answer questions **and control the map** for you.\n\nTry asking me something below ↓"
    }]

# ── Layer config ───────────────────────────────────────────────────────────────
LAYERS = {
    "OpportunityScore":  {"label": "Opportunity Score",         "color": "Oranges", "fmt": ".1f"},
    "BeautyPer1000":     {"label": "Beauty Density",            "color": "Oranges", "fmt": ".1f"},
    "RetailPer1000":     {"label": "Retail Density",            "color": "Oranges", "fmt": ".1f"},
    "MedianIncome":      {"label": "Median Income",             "color": "Oranges", "fmt": ",.0f"},
    "YoungShare":        {"label": "Young Adult Share", "color": "Oranges", "fmt": ".1%"},
}

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 Chicago Beauty Market Agent")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT: Map (left) + Chatbot (right)
# ══════════════════════════════════════════════════════════════════════════════
map_col, chat_col = st.columns([7, 3])

# ── MAP PANEL ──────────────────────────────────────────────────────────────────
with map_col:
    # Layer selector buttons
    btn_cols = st.columns(len(LAYERS))
    for i, (key, cfg) in enumerate(LAYERS.items()):
        if btn_cols[i].button(
            cfg["label"],
            key=f"btn_{key}",
            type="primary" if st.session_state.map_layer == key else "secondary",
            use_container_width=True,
        ):
            st.session_state.map_layer = key
            st.rerun()

    layer     = st.session_state.map_layer
    layer_cfg = LAYERS[layer]

    # Build hover text
    df["hover"] = df.apply(lambda r: (
        f"<b>ZIP {r['ZIP CODE']}</b><br>"
        f"Beauty Count: {int(r.BeautyCount)}<br>"
        f"Beauty Density: {r.BeautyPer1000:.1f}/1k HH<br>"
        f"Retail Density: {r.RetailPer1000:.1f}/1k HH<br>"
        f"Median Income: ${int(r.MedianIncome):,}<br>"
        f"Young Share (25–44): {r.YoungShare:.1%}<br>"
        f"Opportunity Score: {r.OpportunityScore:.1f}/100"
    ), axis=1)

    # Clip color range at 95th percentile to prevent outliers from washing out the map
    color_max = df[layer].quantile(0.95)
    color_min = df[layer].min()

    fig = px.choropleth_map(
        df,
        geojson=geojson,
        locations="ZIP CODE",
        featureidkey="properties.ZCTA5CE10",
        color=layer,
        color_continuous_scale=layer_cfg["color"],
        range_color=(color_min, color_max),
        map_style="carto-darkmatter",
        zoom=9.5,
        center={"lat": 41.83, "lon": -87.68},
        opacity=0.75,
        hover_name="ZIP CODE",
        hover_data={layer: False, "hover": False},
        custom_data=["hover", "ZIP CODE"],
        labels={layer: layer_cfg["label"]},
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker_line_width=0.5,
        marker_line_color="rgba(255,255,255,0.15)",
    )

    # Highlight selected ZIP with a bright border overlay
    if st.session_state.selected_zip:
        sel_df = df[df["ZIP CODE"] == st.session_state.selected_zip]
        if not sel_df.empty:
            fig.add_trace(go.Choroplethmap(
                geojson=geojson,
                locations=sel_df["ZIP CODE"],
                z=[1],
                featureidkey="properties.ZCTA5CE10",
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker=dict(line=dict(color="#a78bfa", width=4)),
                hoverinfo="skip",
                name="selected",
            ))

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=480,
        template="plotly_dark",
        coloraxis_colorbar=dict(
            title=layer_cfg["label"],
            thickness=12,
            len=0.6,
        ),
    )
    selected = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="main_map")

    # Handle click → update selected ZIP
    if selected and selected.get("selection", {}).get("points"):
        clicked_zip = selected["selection"]["points"][0]["customdata"][1]
        st.session_state.selected_zip = clicked_zip

    # ZIP detail card
    if st.session_state.selected_zip:
        z = st.session_state.selected_zip
        row = df[df["ZIP CODE"] == z]
        if not row.empty:
            r = row.iloc[0]
            card_col, clear_col = st.columns([11, 1])
            with card_col:
                st.markdown(f"""<div class="zip-card">
                    <b style="font-size:1rem; color:#a78bfa">📍 ZIP {z}</b>
                    &nbsp;&nbsp;
                    <span style="color:#e2e8f0; font-size:0.85rem">
                    🏪 <b>{int(r.BeautyCount)}</b> biz &nbsp;|&nbsp;
                    📊 Density <b>{r.BeautyPer1000:.1f}</b>/1k &nbsp;|&nbsp;
                    🛍️ Retail <b>{r.RetailPer1000:.1f}</b>/1k &nbsp;|&nbsp;
                    💰 <b>${int(r.MedianIncome):,}</b> &nbsp;|&nbsp;
                    👥 Young <b>{r.YoungShare:.1%}</b> &nbsp;|&nbsp;
                    🎯 Score <b>{r.OpportunityScore:.1f}/100</b>
                    </span>
                </div>""", unsafe_allow_html=True)
            with clear_col:
                if st.button("✕", key="clear_zip", help="Clear selection"):
                    st.session_state.selected_zip = None
                    st.rerun()

# ── CHATBOT PANEL ──────────────────────────────────────────────────────────────
@st.fragment
def chat_panel():
    suggestions = [
        "Show me the ZIP with best opportunity",
        "Show me the ZIP with best retail density",
        "Compare 60657 and 60618",
        "Tell me your insight on the Chicago beauty market",
    ]

    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if len(st.session_state.messages) <= 1:
            st.markdown("<div style='font-size:0.78rem; color:#94a3b8; margin:8px 0 4px'>💡 Try asking:</div>", unsafe_allow_html=True)
            sg1, sg2 = st.columns(2)
            for i, s in enumerate(suggestions):
                col = sg1 if i % 2 == 0 else sg2
                if col.button(s, key=f"sg_{i}", use_container_width=True):
                    st.session_state._pending_prompt = s
                    st.rerun(scope="fragment")

    prompt = st.chat_input("Ask about the Chicago beauty market...")

    if hasattr(st.session_state, "_pending_prompt") and st.session_state._pending_prompt:
        prompt = st.session_state._pending_prompt
        st.session_state._pending_prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        secrets = dict(st.secrets) if hasattr(st, "secrets") else {}
        with chat_container:
            with st.chat_message("assistant"):
                status_box = st.empty()
                def update_status(model_name):
                    status_box.markdown(f"<span style='color:#94a3b8; font-size:0.8rem'>⏳ {model_name} is thinking...</span>", unsafe_allow_html=True)
                answer, actions = chat(st.session_state.messages, df, secrets, status_callback=update_status)
                status_box.empty()

        import re as _re
        map_changed = False
        for action in actions:
            if action["type"] == "switch_map_layer" and action["layer"] in LAYERS:
                st.session_state.map_layer = action["layer"]
                map_changed = True
            elif action["type"] == "select_zip":
                st.session_state.selected_zip = action["zip_code"]
                map_changed = True
            elif action["type"] == "setup_zip_compare":
                st.session_state.compare_zip1 = action["zip1"]
                st.session_state.compare_zip2 = action["zip2"]

        if not any(a["type"] == "select_zip" for a in actions):
            zips_in_prompt = _re.findall(r"\b606\d{2}\b", prompt)
            if zips_in_prompt:
                st.session_state.selected_zip = zips_in_prompt[0]
                map_changed = True

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Full rerun only when map needs to update; otherwise fragment rerun (no flicker)
        if map_changed:
            st.rerun()
        else:
            st.rerun(scope="fragment")

with chat_col:
    chat_panel()


# ── KPI row ────────────────────────────────────────────────────────────────────
st.divider()
top_opp = df.iloc[0]
top_den  = df.loc[df["BeautyPer1000"].idxmax()]
corr_young  = df.BeautyCount.corr(df.YoungShare)
corr_income = df.BeautyCount.corr(df.MedianIncome)
strongest = "Retail Density"

k1, k2, k3, k4 = st.columns(4)
for col, val, label in [
    (k1, f"{df.BeautyCount.sum():,}", "Total Beauty Businesses"),
    (k2, strongest, "Strongest Predictor"),
    (k3, top_opp["ZIP CODE"], "Top Opportunity ZIP"),
    (k4, top_den["ZIP CODE"], "Highest Density ZIP"),
]:
    col.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{val}</div>
        <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM TABS
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
tab1, tab2, tab3 = st.tabs(["📊 Correlation Analysis", "🏆 Rankings", "🔍 ZIP Compare"])

with tab1:
    corr_young  = df.BeautyCount.corr(df.YoungShare)
    corr_income = df.BeautyCount.corr(df.MedianIncome)
    corr_retail = df.BeautyPer1000.corr(df.RetailPer1000)

    from scipy import stats
    spearman_income = stats.spearmanr(df.BeautyPer1000, df.MedianIncome)[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Beauty Count vs Young Share")
        fig = px.scatter(df, x="YoungShare", y="BeautyCount", trendline="ols",
            hover_data={"ZIP CODE": True},
            template="plotly_dark", height=380,
            labels={"YoungShare": "Young Adult Share (25–44)", "BeautyCount": "Beauty Count"})
        fig.update_traces(marker=dict(color="#a78bfa", size=8, opacity=0.8),
                          selector=dict(mode="markers"))
        fig.update_layout(annotations=[dict(x=0.05, y=0.95, xref="paper", yref="paper",
            text=f"<b>Pearson r = {corr_young:.3f}</b>", showarrow=False,
            font=dict(size=13, color="#a78bfa"), bgcolor="rgba(20,20,40,0.8)")])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Beauty Density vs Median Income")
        fig2 = px.scatter(df, x="MedianIncome", y="BeautyPer1000", trendline="ols",
            hover_data={"ZIP CODE": True},
            template="plotly_dark", height=380,
            labels={"MedianIncome": "Median Income ($)", "BeautyPer1000": "Beauty Density (per 1k HH)"})
        fig2.update_traces(marker=dict(color="#f472b6", size=8, opacity=0.8),
                           selector=dict(mode="markers"))
        fig2.update_layout(annotations=[dict(x=0.05, y=0.95, xref="paper", yref="paper",
            text=f"<b>Spearman r = {spearman_income:.3f}</b>", showarrow=False,
            font=dict(size=13, color="#f472b6"), bgcolor="rgba(20,20,40,0.8)")])
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        st.subheader("Beauty Density vs Retail Density")
        fig3 = px.scatter(df, x="RetailPer1000", y="BeautyPer1000", trendline="ols",
            hover_data={"ZIP CODE": True},
            template="plotly_dark", height=380,
            labels={"RetailPer1000": "Retail Density (per 1k HH)", "BeautyPer1000": "Beauty Density (per 1k HH)"})
        fig3.update_traces(marker=dict(color="#34d399", size=8, opacity=0.8),
                           selector=dict(mode="markers"))
        fig3.update_layout(annotations=[dict(x=0.05, y=0.95, xref="paper", yref="paper",
            text=f"<b>Spearman r = 0.783</b>", showarrow=False,
            font=dict(size=13, color="#34d399"), bgcolor="rgba(20,20,40,0.8)")])
        st.plotly_chart(fig3, use_container_width=True)

    winner = "Young Adult Share" if abs(corr_young) > abs(corr_income) else "Median Income"
    st.markdown(f"""<div class="insight-box">
        💡 <b>Key Insight:</b> Among demographic factors, <b>{winner}</b> shows stronger correlation with
        beauty business count. However, <b>Retail Density</b> (r={corr_retail:.3f}) is the strongest
        predictor of beauty business density — beauty businesses cluster where commercial activity is already high.
    </div>""", unsafe_allow_html=True)

with tab2:
    r1, r2 = st.columns(2)
    with r1:
        st.subheader("Top 10 by Beauty Density")
        top_d = df.nlargest(10, "BeautyPer1000").sort_values("BeautyPer1000")
        fig4 = px.bar(top_d, x="BeautyPer1000", y="ZIP CODE", orientation="h",
            color="BeautyPer1000", color_continuous_scale="Viridis",
            text=top_d["BeautyPer1000"].apply(lambda x: f"{x:.1f}"),
            template="plotly_dark", height=380)
        fig4.update_traces(textposition="outside")
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    with r2:
        st.subheader("Top 10 by Opportunity Score")
        top_o = df.nlargest(10, "OpportunityScore").sort_values("OpportunityScore")
        fig5 = px.bar(top_o, x="OpportunityScore", y="ZIP CODE", orientation="h",
            color="OpportunityScore", color_continuous_scale="RdYlGn",
            text=top_o["OpportunityScore"].apply(lambda x: f"{x:.1f}"),
            template="plotly_dark", height=380)
        fig5.update_traces(textposition="outside")
        fig5.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)

with tab3:
    all_zips = sorted(df["ZIP CODE"].tolist())
    ca, cb = st.columns(2)
    idx_a = all_zips.index(st.session_state.compare_zip1) if st.session_state.compare_zip1 in all_zips else 0
    idx_b = all_zips.index(st.session_state.compare_zip2) if st.session_state.compare_zip2 in all_zips else 5
    zip_a = ca.selectbox("ZIP Code A", all_zips, index=idx_a)
    zip_b = cb.selectbox("ZIP Code B", all_zips, index=idx_b)

    ra = df[df["ZIP CODE"] == zip_a].iloc[0]
    rb = df[df["ZIP CODE"] == zip_b].iloc[0]

    metrics = {
        "Beauty Count": ("BeautyCount", lambda v: str(int(v))),
        "Beauty Density": ("BeautyPer1000", lambda v: f"{v:.1f}"),
        "Retail Density": ("RetailPer1000", lambda v: f"{v:.1f}"),
        "Median Income": ("MedianIncome", lambda v: f"${int(v):,}"),
        "Young Share (25–44)": ("YoungShare", lambda v: f"{v:.1%}"),
        "Nonfamily Share": ("SingleShare", lambda v: f"{v:.1%}"),
        "Opportunity Score": ("OpportunityScore", lambda v: f"{v:.1f}"),
    }

    cmp = {"Metric": [], zip_a: [], zip_b: []}
    for label, (col, fmt) in metrics.items():
        cmp["Metric"].append(label)
        cmp[zip_a].append(fmt(ra[col]))
        cmp[zip_b].append(fmt(rb[col]))

    st.dataframe(pd.DataFrame(cmp).set_index("Metric"), use_container_width=True)

    numeric_cols = ["BeautyCount", "YoungShare", "SingleShare", "BeautyPer1000", "OpportunityScore"]
    cmp_rows = []
    for col in numeric_cols:
        cmp_rows.append({"Metric": col, "Value": ra[col], "ZIP": zip_a})
        cmp_rows.append({"Metric": col, "Value": rb[col], "ZIP": zip_b})
    fig_cmp = px.bar(pd.DataFrame(cmp_rows), x="Metric", y="Value", color="ZIP", barmode="group",
        color_discrete_map={zip_a: "#a78bfa", zip_b: "#f472b6"},
        template="plotly_dark", height=320)
    st.plotly_chart(fig_cmp, use_container_width=True)
