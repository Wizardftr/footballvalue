"""Visual identity for the dashboard.

The palette avoids the obvious choice. Betting products default to green-on-black,
which reads as "money" and quietly promises profit; this app's central claim is
that profit is *not* established. So the accent is a cool teal used for navigation
and focus, green and red appear only where they mean a settled win or loss, and
nothing about the chrome congratulates you.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

WORDMARK = Path(__file__).with_name("assets") / "wordmark.svg"

INK = "#0E141B"
SURFACE = "#171F29"
LINE = "#26303C"
LINE_BRIGHT = "#37455A"
TEXT = "#E4EBF2"
MUTED = "#8A99AB"
ACCENT = "#4FB3A9"
WARN = "#E0A458"
BAD = "#D9636A"
GOOD = "#5FB98A"

CSS = f"""
<style>
  /* Streamlit's default chrome reads as a notebook. Strip it back to an app. */
  #MainMenu, header [data-testid="stStatusWidget"] {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  .block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1120px; }}

  html, body, [class*="css"] {{
    /* Tabular figures: money and odds must line up column to column. */
    font-feature-settings: "tnum" 1, "cv05" 1;
    -webkit-font-smoothing: antialiased;
  }}

  /* Typographic scale. Streamlit's defaults are sized for notebooks; these are
     sized for reading one screen at a time. */
  h1, h2, h3 {{ letter-spacing: -.022em; }}
  h2 {{ font-size: 1.28rem !important; font-weight: 640 !important;
        margin: 2.4rem 0 .2rem 0 !important; }}
  h3 {{ font-size: 1.02rem !important; font-weight: 620 !important;
        margin: 1.6rem 0 .2rem 0 !important; }}
  [data-testid="stCaptionContainer"] p {{ color: {MUTED}; font-size: .82rem; }}

  /* Wordmark */
  .fv-brand {{
    display: flex; align-items: baseline; gap: .1rem;
    font-size: 1.42rem; font-weight: 700; letter-spacing: -.02em;
  }}
  .fv-brand .a {{ color: {TEXT}; }}
  .fv-brand .b {{ color: {ACCENT}; }}
  .fv-tag {{ color: {MUTED}; font-size: .76rem; margin-bottom: 1.1rem; }}

  /* Page heading */
  .fv-hero h1 {{
    font-size: 2.1rem; font-weight: 680; letter-spacing: -.028em;
    margin: 0 0 .3rem 0; line-height: 1.12;
  }}
  .fv-hero p {{ color: {MUTED}; font-size: .94rem; margin: 0 0 1.6rem 0; max-width: 62ch;
                line-height: 1.55; }}

  /* Stat cards */
  .fv-card {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px;
    padding: 1.05rem 1.15rem; margin-bottom: .8rem; height: 100%;
    transition: border-color .15s ease;
  }}
  .fv-card:hover {{ border-color: {LINE_BRIGHT}; }}
  .fv-card .k {{
    color: {MUTED}; font-size: .72rem; text-transform: uppercase;
    letter-spacing: .08em; font-weight: 600; margin-bottom: .4rem;
  }}
  .fv-card .v {{ font-size: 1.72rem; font-weight: 650; letter-spacing: -.03em;
                 line-height: 1.05; }}
  .fv-card .s {{ color: {MUTED}; font-size: .78rem; margin-top: .35rem; line-height: 1.45; }}
  .fv-card.accent {{ border-left: 3px solid {ACCENT}; }}
  .fv-card.warn   {{ border-left: 3px solid {WARN}; }}
  .fv-card.bad    {{ border-left: 3px solid {BAD}; }}
  .fv-card.good   {{ border-left: 3px solid {GOOD}; }}

  /* Day heading above a group of fixtures */
  .fv-day {{
    display: flex; align-items: baseline; gap: .6rem;
    margin: 1.5rem 0 .5rem 0; padding-bottom: .4rem;
    border-bottom: 1px solid {LINE};
  }}
  .fv-day .d {{ font-weight: 640; font-size: .95rem; letter-spacing: -.01em; }}
  .fv-day .n {{ color: {MUTED}; font-size: .78rem; }}

  /* Tables */
  [data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 12px;
                                 overflow: hidden; }}

  /* Buttons */
  .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    border-radius: 9px; font-weight: 570; letter-spacing: -.005em;
  }}

  /* Inputs */
  [data-baseweb="input"], [data-baseweb="select"] > div {{ border-radius: 9px; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
  [data-testid="stSidebarNav"] a {{ border-radius: 8px; }}

  /* Expanders read as secondary, not as headings */
  [data-testid="stExpander"] {{ border: 1px solid {LINE}; border-radius: 12px; }}
  [data-testid="stExpander"] summary p {{ font-weight: 560; font-size: .9rem; }}

  /* The disclaimer that has to be present and must not look decorative */
  .fv-legal {{
    color: {MUTED}; font-size: .74rem; line-height: 1.55;
    border-top: 1px solid {LINE}; padding-top: .9rem; margin-top: 3rem;
  }}

  /* Narrow screens: the sidebar collapses, so the content needs its own room. */
  @media (max-width: 640px) {{
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    .fv-hero h1 {{ font-size: 1.6rem; }}
    .fv-card .v {{ font-size: 1.4rem; }}
  }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    # st.logo puts the mark above the navigation, which is the one place Streamlit
    # will not let ordinary sidebar content reach.
    if WORDMARK.exists():
        st.logo(str(WORDMARK), size="large")


def wordmark(container=st) -> None:
    container.markdown(
        '<div class="fv-brand"><span class="a">football</span><span class="b">value</span></div>'
        '<div class="fv-tag">Analysis only &middot; it never places bets</div>',
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="fv-hero"><h1>{title}</h1>{sub}</div>', unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", tone: str = "", container=st) -> None:
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    container.markdown(
        f'<div class="fv-card {tone}"><div class="k">{label}</div>'
        f'<div class="v">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def legal_footer() -> None:
    """Present on every page, deliberately plain.

    Two claims this app must never let its own UI imply: that the selections are
    profitable, and that gambling is a way to make money.
    """
    st.markdown(
        '<div class="fv-legal">'
        "18+ only. This is an analysis tool, not betting advice, and it does not place "
        "bets. Its own backtest does not beat bet365&rsquo;s closing prices, so treat "
        "every selection as unproven and stake only what you can afford to lose. "
        "Gambling is addictive &mdash; support is available at "
        "<a href='https://www.begambleaware.org' target='_blank'>BeGambleAware.org</a>."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

@alt.theme.register("footballvalue", enable=True)
def _altair_theme() -> alt.theme.ThemeConfig:
    """One chart style for the whole app.

    Streamlit's built-in charts are fine for a notebook and wrong here: they pick
    their own colours, draw a heavy grid, and label axes with the dataframe's
    column names. Registering a theme once means every chart in the app agrees,
    and money always renders in the same colour as money.
    """
    return alt.theme.ThemeConfig({
        "config": {
            "background": "transparent",
            "font": "-apple-system, BlinkMacSystemFont, Inter, Helvetica, sans-serif",
            "view": {"stroke": "transparent", "continuousHeight": 260},
            "axis": {
                "labelColor": MUTED, "titleColor": MUTED,
                "labelFontSize": 11, "titleFontSize": 11, "titleFontWeight": 500,
                "domainColor": LINE, "tickColor": LINE,
                "gridColor": LINE, "gridOpacity": 0.5, "gridDash": [2, 3],
            },
            "legend": {"labelColor": TEXT, "titleColor": MUTED, "labelFontSize": 11,
                       "titleFontSize": 11, "symbolType": "stroke"},
            "range": {"category": [ACCENT, WARN, BAD, GOOD, MUTED]},
            "line": {"strokeWidth": 2},
            "bar": {"color": ACCENT, "cornerRadiusEnd": 3},
            "area": {"line": True, "opacity": 0.18},
        }
    })


def line_chart(df: pd.DataFrame, x: str, y: str, y_title: str = "",
               zero_line: bool = False, height: int = 260):
    """A single series over time, with an optional break-even rule at zero."""
    base = alt.Chart(df).mark_line(color=ACCENT).encode(
        x=alt.X(f"{x}:T", title=None),
        y=alt.Y(f"{y}:Q", title=y_title or None),
        tooltip=[alt.Tooltip(f"{x}:T", title="Date"),
                 alt.Tooltip(f"{y}:Q", title=y_title or y, format=",.2f")],
    ).properties(height=height)
    if not zero_line:
        return base
    rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=MUTED, strokeDash=[4, 4], opacity=0.7
    ).encode(y="y:Q")
    return rule + base


def band_chart(df: pd.DataFrame, x: str, y: str, lo: str, hi: str,
               y_title: str = "", height: int = 280):
    """A line inside its uncertainty band.

    The band is the point. A rolling return drawn as a bare line invites exactly
    the misreading this project exists to avoid, so the interval is drawn first and
    the line sits inside it rather than the other way round.
    """
    area = alt.Chart(df).mark_area(opacity=0.16, color=ACCENT).encode(
        x=alt.X(f"{x}:T", title=None),
        y=alt.Y(f"{lo}:Q", title=y_title or None),
        y2=alt.Y2(f"{hi}:Q"),
    )
    line = alt.Chart(df).mark_line(color=ACCENT, strokeWidth=2).encode(
        x=f"{x}:T", y=f"{y}:Q",
        tooltip=[alt.Tooltip(f"{x}:T", title="Date"),
                 alt.Tooltip(f"{y}:Q", title="Return", format="+.1%"),
                 alt.Tooltip(f"{lo}:Q", title="Could be as low as", format="+.1%"),
                 alt.Tooltip(f"{hi}:Q", title="Could be as high as", format="+.1%")],
    )
    rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=MUTED, strokeDash=[4, 4], opacity=0.7
    ).encode(y="y:Q")
    return (area + rule + line).properties(height=height)


def bar_chart(df: pd.DataFrame, x: str, y: str, y_title: str = "", height: int = 240):
    """Profit by period. Losing bars are coloured as losses, not as data."""
    return alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X(f"{x}:N", title=None, sort=None),
        y=alt.Y(f"{y}:Q", title=y_title or None),
        color=alt.condition(alt.datum[y] < 0, alt.value(BAD), alt.value(GOOD)),
        tooltip=[alt.Tooltip(f"{x}:N", title="Month"),
                 alt.Tooltip(f"{y}:Q", title="Profit", format=",.2f")],
    ).properties(height=height)


def day_heading(label: str, count: int) -> None:
    st.markdown(
        f'<div class="fv-day"><span class="d">{label}</span>'
        f'<span class="n">{count} pick{"" if count == 1 else "s"}</span></div>',
        unsafe_allow_html=True,
    )
