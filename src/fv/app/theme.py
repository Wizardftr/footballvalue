"""Visual identity for the dashboard.

The palette avoids the obvious choice. Betting products default to green-on-black,
which reads as "money" and quietly promises profit; this app's central claim is
that profit is *not* established. So the accent is a cool teal used for navigation
and focus, green and red appear only where they mean a settled win or loss, and
nothing about the chrome congratulates you.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

WORDMARK = Path(__file__).with_name("assets") / "wordmark.svg"

INK = "#0E141B"
SURFACE = "#171F29"
LINE = "#26303C"
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
  .block-container {{ padding-top: 2.4rem; max-width: 1180px; }}

  html, body, [class*="css"] {{
    font-feature-settings: "tnum" 1, "cv05" 1;
  }}

  /* Wordmark */
  .fv-brand {{
    display: flex; align-items: baseline; gap: .1rem;
    font-size: 1.42rem; font-weight: 700; letter-spacing: -.02em;
    margin: .1rem 0 .1rem 0;
  }}
  .fv-brand .a {{ color: {TEXT}; }}
  .fv-brand .b {{ color: {ACCENT}; }}
  .fv-tag {{
    color: {MUTED}; font-size: .76rem; letter-spacing: .02em;
    margin-bottom: 1.1rem;
  }}

  /* Page heading */
  .fv-hero h1 {{
    font-size: 2.05rem; font-weight: 680; letter-spacing: -.025em;
    margin: 0 0 .25rem 0; line-height: 1.15;
  }}
  .fv-hero p {{ color: {MUTED}; font-size: .93rem; margin: 0 0 1.4rem 0; max-width: 62ch; }}

  /* Cards */
  .fv-card {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px;
    padding: 1.05rem 1.2rem; margin-bottom: .85rem;
  }}
  .fv-card .k {{
    color: {MUTED}; font-size: .74rem; text-transform: uppercase;
    letter-spacing: .07em; margin-bottom: .35rem;
  }}
  .fv-card .v {{ font-size: 1.7rem; font-weight: 640; letter-spacing: -.02em; line-height: 1.1; }}
  .fv-card .s {{ color: {MUTED}; font-size: .78rem; margin-top: .3rem; }}
  .fv-card.accent {{ border-left: 3px solid {ACCENT}; }}
  .fv-card.warn   {{ border-left: 3px solid {WARN}; }}
  .fv-card.bad    {{ border-left: 3px solid {BAD}; }}

  /* Tables: quieter borders, tabular figures */
  [data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 10px; }}

  /* Buttons */
  .stButton > button {{ border-radius: 8px; font-weight: 560; }}

  /* The disclaimer that has to be present and must not look decorative */
  .fv-legal {{
    color: {MUTED}; font-size: .74rem; line-height: 1.5;
    border-top: 1px solid {LINE}; padding-top: .9rem; margin-top: 2.6rem;
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
