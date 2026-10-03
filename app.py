"""GapDetect Streamlit app — conversational communication-gap analysis.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import DEFAULT_CONFIG, DetectorConfig
from core.engine import EvidenceEngine
from core.parser import parse_any, parse_file
from core.schema import Gap, Message

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

GAP_COLORS = {
    "unanswered": "#FF6B6B",
    "ignored": "#FFA94D",
    "clarification": "#4DABF7",
    "unresolved": "#B197FC",
    "error": "#ADB5BD",
}
GAP_ICONS = {
    "unanswered": "❓",
    "ignored": "🚫",
    "clarification": "💬",
    "unresolved": "🧩",
}

st.set_page_config(
    page_title="GapDetect — Communication Gap Analyzer",
    page_icon="🔍",
    layout="wide",
)


def severity_badge(sev: float) -> str:
    if sev >= 0.7:
        return f"🔴 High ({sev:.2f})"
    if sev >= 0.5:
        return f"🟡 Medium ({sev:.2f})"
    return f"🔵 Low ({sev:.2f})"


def gap_title(gap_type: str) -> str:
    return gap_type.replace("_", " ").title()


@st.cache_data(show_spinner=False)
def load_sample(path_str: str) -> str:
    p = Path(path_str)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def main() -> None:
    st.title("🔍 GapDetect — Communication Gap Analyzer")
    st.caption(
        "Finds unanswered questions, ignored addresses, repeated clarification asks, "
        "and unresolved topics — with exact message evidence."
    )

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    st.sidebar.header("📁 Data Source")
    mode = st.sidebar.radio(
        "Source",
        ["Sample chat", "Upload file", "Paste text"],
        label_visibility="collapsed",
    )

    raw_text = ""
    source_name = ""

    if mode == "Sample chat":
        samples = sorted(DATA_DIR.glob("*.txt"))
        labels = {p.stem: p for p in DATA_DIR.glob("*.labels.json")}
        options = [p.stem for p in samples] or ["(none)"]
        choice = st.sidebar.selectbox("Sample", options)
        if choice != "(none)":
            raw_text = load_sample(str(DATA_DIR / f"{choice}.txt"))
            source_name = choice
            if choice in labels:
                st.sidebar.success(f"Labeled: data/{choice}.labels.json")
    elif mode == "Upload file":
        up = st.sidebar.file_uploader("WhatsApp .txt / JSON / CSV", type=["txt", "json", "csv"])
        if up is not None:
            raw_text = up.getvalue().decode("utf-8", errors="replace")
            source_name = up.name
    else:
        raw_text = st.sidebar.text_area(
            "Paste conversation",
            height=220,
            placeholder="[12/03/2026, 10:00:15] Alice: Can someone check this?\n[12/03/2026, 10:01:00] Bob: Hey!",
        )
        source_name = "pasted"

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Thresholds")

    cfg = DetectorConfig()
    min_severity = st.sidebar.slider("Min severity", 0.0, 1.0, float(cfg.min_severity), 0.05)
    unanswered_window = st.sidebar.slider(
        "Unanswered lookahead (msgs)", 3, 30, cfg.unanswered_lookahead_messages, 1
    )
    unanswered_minutes = st.sidebar.slider(
        "Unanswered lookahead (min)", 30, 1440, int(cfg.unanswered_lookahead_minutes), 30
    )
    clar_sim = st.sidebar.slider(
        "Clarification similarity", 0.10, 0.90, cfg.clarification_similarity_threshold, 0.05
    )
    ignored_window = st.sidebar.slider(
        "Ignored lookahead (msgs)", 3, 30, cfg.ignored_lookahead_messages, 1
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🔌 Optional features")
    use_embeddings = st.sidebar.checkbox(
        "Sentence-transformer embeddings",
        value=False,
        help="Adds re-ask detection via cosine embeddings if sentence-transformers is installed.",
    )
    use_llm = st.sidebar.checkbox(
        "Gemini verification",
        value=False,
        help="Requires GEMINI_API_KEY. App works fully with this OFF.",
    )
    if use_llm:
        import os

        if not os.environ.get("GEMINI_API_KEY"):
            st.sidebar.warning("GEMINI_API_KEY not set — verification will be skipped.")

    cfg.min_severity = min_severity
    cfg.unanswered_lookahead_messages = unanswered_window
    cfg.unanswered_lookahead_minutes = float(unanswered_minutes)
    cfg.clarification_similarity_threshold = clar_sim
    cfg.ignored_lookahead_messages = ignored_window

    if not raw_text.strip():
        st.info("👈 Choose a sample chat, upload an export, or paste a conversation in the sidebar.")
        return

    # ------------------------------------------------------------------
    # Parse + analyze
    # ------------------------------------------------------------------
    try:
        messages = parse_any(raw_text)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Parse error: {exc}")
        return
    if not messages:
        st.warning("No messages parsed. Check the input format.")
        return

    engine = EvidenceEngine(cfg)
    with st.spinner("Analyzing conversation…"):
        result = engine.analyze(
            messages,
            use_adapters=True,
            use_embeddings=use_embeddings,
            use_llm_verify=use_llm,
        )
    messages = result.messages
    gaps = result.gaps

    # ------------------------------------------------------------------
    # Metric cards
    # ------------------------------------------------------------------
    counts = result.gaps_by_type()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Messages", len(messages))
    c2.metric("Participants", len(result.participants))
    c3.metric("Total gaps", len(gaps))
    c4.metric("Unanswered", counts.get("unanswered", 0))
    c5.metric("Unresolved", counts.get("unresolved", 0))

    c6, c7, c8 = st.columns(3)
    c6.metric("Ignored", counts.get("ignored", 0))
    c7.metric("Clarification", counts.get("clarification", 0))
    avg_sev = sum(g.severity for g in gaps) / len(gaps) if gaps else 0.0
    c8.metric("Avg severity", f"{avg_sev:.2f}")

    if result.adapter_report:
        modes = ", ".join(f"{a.name}:{a.mode}" for a in result.adapter_report.adapters)
        st.caption(f"Adapters — {modes}")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Left: transcript | Right: gap cards
    # ------------------------------------------------------------------
    left, right = st.columns([1.1, 1], gap="large")

    # Map message id -> gap types (for highlighting)
    msg_types: dict[int, set[str]] = {}
    for g in gaps:
        for mid in g.message_ids:
            msg_types.setdefault(mid, set()).add(g.gap_type)

    with left:
        st.subheader("💬 Transcript")
        type_filter = st.multiselect(
            "Highlight types",
            options=list(GAP_COLORS.keys() - {"error"}),
            default=list(GAP_COLORS.keys() - {"error"}),
        )
        filter_sev = st.slider("Min severity to highlight", 0.0, 1.0, min_severity, 0.05)

        highlight_ids = {
            mid
            for g in gaps
            if g.severity >= filter_sev and (not type_filter or g.gap_type in type_filter)
            for mid in g.message_ids
        }

        for m in messages:
            types = msg_types.get(m.id, set())
            flagged = m.id in highlight_ids
            ts = m.timestamp.strftime("%H:%M") if m.timestamp else ""
            addr = f" → @{m.addressee}" if m.addressee and m.addressee != "group" else ""
            header = f"**#{m.id}** `{ts}` **{m.sender}**{addr}"
            body = m.text.replace("\n", " ")
            if flagged and types:
                color = GAP_COLORS.get(sorted(types)[0], "#868E96")
                badges = " ".join(
                    f"<span style='background:{GAP_COLORS.get(t,'#868E96')};color:white;"
                    f"border-radius:8px;padding:1px 7px;margin-right:4px;font-size:0.75em;'>"
                    f"{GAP_ICONS.get(t,'')} {t}</span>"
                    for t in sorted(types)
                )
                st.markdown(
                    f"<div id='msg-{m.id}' style='border-left:5px solid {color};"
                    f"background:{color}22;padding:8px 10px;margin:4px 0;border-radius:6px;'>"
                    f"{header} {badges}<br/>{body}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div id='msg-{m.id}' style='padding:4px 8px;margin:2px 0;'>"
                    f"{header}<br/>{body}</div>",
                    unsafe_allow_html=True,
                )

    with right:
        st.subheader("🚨 Gap Cards")
        if not gaps:
            st.success("No gaps found at the current thresholds.")
        type_order = {"unanswered": 0, "ignored": 1, "clarification": 2, "unresolved": 3}
        sorted_gaps = sorted(
            gaps,
            key=lambda g: (-g.severity, type_order.get(g.gap_type, 9), g.message_ids[0] if g.message_ids else 0),
        )
        for idx, gap in enumerate(sorted_gaps):
            color = GAP_COLORS.get(gap.gap_type, "#868E96")
            icon = GAP_ICONS.get(gap.gap_type, "⚠️")
            with st.container(border=True):
                st.markdown(
                    f"<div style='border-left:5px solid {color};padding-left:8px;'>"
                    f"<b>{icon} {gap_title(gap.gap_type)}</b> &nbsp; {severity_badge(gap.severity)}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(gap.explanation)
                with st.expander("Evidence quotes", expanded=(idx < 3)):
                    for ev in gap.evidence:
                        st.markdown(f"> {ev}")
                    st.caption(f"Message IDs: {gap.message_ids}")
                ids_html = " | ".join(
                    f"<a href='#msg-{mid}' style='color:{color};'>#{mid}</a>" for mid in gap.message_ids
                )
                st.markdown(f"Jump to: {ids_html}", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Analytics")

    # Gaps per participant (who is ignored / left hanging most)
    person_gaps: dict[str, int] = {}
    for g in gaps:
        if g.gap_type not in ("ignored", "unanswered"):
            continue
        for mid in g.message_ids:
            m = next((x for x in messages if x.id == mid), None)
            if m:
                # The addressee is the one left hanging for ignored/unanswered
                key = m.addressee if g.gap_type == "ignored" and m.addressee and m.addressee != "group" else m.sender
                person_gaps[key] = person_gaps.get(key, 0) + 1

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Gaps involving participants**")
        if person_gaps:
            df_p = pd.DataFrame(
                sorted(person_gaps.items(), key=lambda x: -x[1]), columns=["Participant", "Gaps"]
            )
            st.bar_chart(df_p.set_index("Participant"))
        else:
            st.write("No participant-level gaps.")

    with ch2:
        st.markdown("**Gaps over time**")
        rows = []
        for g in gaps:
            for mid in g.message_ids:
                m = next((x for x in messages if x.id == mid), None)
                if m:
                    rows.append({"time": m.timestamp, "gap_type": g.gap_type})
        if rows:
            df_t = pd.DataFrame(rows)
            df_t["hour"] = df_t["time"].dt.floor("h")
            pivot = (
                df_t.groupby(["hour", "gap_type"]).size().unstack(fill_value=0)
            )
            st.line_chart(pivot)
        else:
            st.write("No timed gaps.")

    by_type_df = pd.DataFrame(
        [{"Gap type": gap_title(k), "Count": v} for k, v in sorted(counts.items())]
    )
    if not by_type_df.empty:
        st.markdown("**Gaps by type**")
        st.bar_chart(by_type_df.set_index("Gap type"))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Export")
    export = result.to_json()
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download JSON",
            data=export,
            file_name=f"gapdetect_{source_name or 'chat'}.json",
            mime="application/json",
        )
    with e2:
        csv_buf = io.StringIO()
        csv_buf.write("gap_id,gap_type,severity,message_ids,explanation\n")
        for i, g in enumerate(gaps, start=1):
            exp = g.explanation.replace('"', "'").replace("\n", " ")
            csv_buf.write(
                f"{i},{g.gap_type},{g.severity:.3f},"
                f"'{';'.join(str(x) for x in g.message_ids)}','{exp}'\n"
            )
        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"gapdetect_{source_name or 'chat'}.csv",
            mime="text/csv",
        )

    with st.expander("Raw JSON preview"):
        st.code(export[:4000] + ("\n…" if len(export) > 4000 else ""), language="json")


if __name__ == "__main__":
    main()
