"""Streamlit entry point for the pipeline profiler."""

import streamlit as st

from telemetry.profiler.tempo import TempoError, list_profiles, load_profile
from telemetry.profiler.trace import InvalidProfileError
from telemetry.profiler.view import render_profile


def main() -> None:
    st.set_page_config(page_title="Pipeline Profiler", layout="wide")
    st.title("Pipeline Profiler")
    st.caption("Completed request profiles stored in Tempo")

    if st.sidebar.button("Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    try:
        profiles = list_profiles()
    except TempoError as error:
        st.error(str(error))
        st.info("Start Tempo with `docker compose up -d` and refresh this page.")
        st.stop()

    if not profiles:
        st.info("No completed pipeline requests were found in Tempo.")
        st.stop()

    profile_labels = {profile.trace_id: profile.label for profile in profiles}
    selected_trace_id = st.sidebar.selectbox(
        "Run",
        [profile.trace_id for profile in profiles],
        format_func=lambda trace_id: profile_labels[trace_id],
    )
    try:
        render_profile(load_profile(selected_trace_id))
    except (TempoError, InvalidProfileError) as error:
        st.error(str(error))


if __name__ == "__main__":
    main()
