import streamlit as st
from modules.calculations.interpretation import generate_training_advice

def render_coach_summary(metrics: dict, quality_report: dict = None):
    """
    Render a high-level coaching summary.
    
    Args:
        metrics: Dict with key metrics (VT1, VT2, Tau, etc.)
        quality_report: Output from check_data_quality
    """
    if quality_report is None:
        quality_report = {"is_valid": True, "issues": []}
        
    advice = generate_training_advice(metrics, quality_report)
    
    st.header("🧠 Automated Coach (Wnioski Treningowe)")
    
    # 1. Status Section
    if not advice['is_valid']:
        st.error("🚫 Brak Zaleceń: Dane są niewystarczającej jakości.")
        with st.expander("Szczegóły ostrzeżeń"):
            for w in advice['warnings']:
                st.markdown(f"- {w}")
        return

    # 2. Diagnostics Section
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🔍 Diagnoza Fizjologiczna")
        if advice['diagnostics']:
            for diag in advice['diagnostics']:
                st.info(f"**Obserwacja:** {diag}")
        else:
            st.success("Brak wyraźnych deficytów.")
            
    with c2:
        st.subheader("📋 Sugestie Treningowe")
        if advice['prescriptions']:
            for presc in advice['prescriptions']:
                st.markdown(f"✅ **Zalecenie:** {presc}")
        else:
            st.markdown("Kontynuuj obecny plan.")
            
    # 3. Warnings (if valid but with caveats)
    if advice['warnings']:
        st.caption("⚠️ Uwagi dodatkowe:")
        for w in advice['warnings']:
            st.caption(f"- {w}")
