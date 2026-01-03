"""
HTML Report Generator for Ramp Test.

Converts canonical JSON results into a client-facing HTML document.
Focus:
- Non-medical language
- Visual confidence representation
- Explicit limitations
- Professional layout
"""
from typing import Dict, Any, List
from datetime import datetime

# ============================================================
# CSS STYLES
# ============================================================

CSS = """
<style>
    body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; background: #f9f9f9; }
    .container { background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }
    h3 { color: #555; margin-top: 20px; }
    .meta { color: #7f8c8d; font-size: 0.9em; margin-bottom: 30px; }
    
    /* Validity Badges */
    .badge { padding: 5px 10px; border-radius: 4px; color: #fff; font-weight: bold; display: inline-block; }
    .valid { background: #27ae60; }
    .conditional { background: #f39c12; }
    .invalid { background: #c0392b; }
    
    /* Threshold Cards */
    .card { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 20px; margin-bottom: 15px; }
    .range-display { font-size: 1.4em; font-weight: bold; color: #2c3e50; }
    .sub-text { color: #7f8c8d; font-size: 0.9em; }
    
    /* Confidence Bar */
    .confidence-wrapper { margin-top: 10px; background: #eee; height: 10px; border-radius: 5px; width: 100%; overflow: hidden; }
    .confidence-fill { height: 100%; transition: width 0.3s; }
    .conf-high { background: #27ae60; }
    .conf-medium { background: #f39c12; }
    .conf-low { background: #e74c3c; }
    
    /* Notes & Warnings */
    .note { background: #e8f6f3; border-left: 4px solid #1abc9c; padding: 10px; margin: 10px 0; font-size: 0.95em; }
    .warning { background: #fef9e7; border-left: 4px solid #f1c40f; padding: 10px; margin: 10px 0; font-size: 0.95em; }
    .conflict { background: #fdedec; border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; }
    
    /* Disclaimer */
    .disclaimer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; font-size: 0.8em; color: #95a5a6; text-align: center; }
    
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { text-align: left; padding: 10px; border-bottom: 1px solid #eee; }
    th { background-color: #f8f9fa; font-weight: 600; }
</style>
"""

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _confidence_color(level: str) -> str:
    if level == "high": return "conf-high"
    if level == "medium": return "conf-medium"
    return "conf-low"

def _format_confidence(conf: float) -> str:
    percent = int(conf * 100)
    level = "high" if conf >= 0.8 else ("medium" if conf >= 0.5 else "low")
    return f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
        <span style="font-size: 0.85em; color: #666;">Pewność detekcji:</span>
        <span style="font-weight: bold; font-size: 0.9em;">{percent}% ({_translate_level(level)})</span>
    </div>
    <div class="confidence-wrapper">
        <div class="confidence-fill {_confidence_color(level)}" style="width: {percent}%"></div>
    </div>
    """

def _translate_level(level: str) -> str:
    mapping = {"high": "wysoka", "medium": "średnia", "low": "niska"}
    return mapping.get(level, level)

# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_html_report(data: Dict[str, Any]) -> str:
    """
    Generate complete HTML report from canonical JSON dictionary.
    """
    meta = data.get("metadata", {})
    interp = data.get("interpretation", {})
    thresholds = data.get("thresholds", {})
    validity = data.get("test_validity", {})
    conflicts = data.get("conflicts", {})
    
    # Header
    html = f"""
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <title>Raport Ramp Test</title>
        {CSS}
    </head>
    <body>
        <div class="container">
            <h1>Raport Wydolnościowy: Ramp Test</h1>
            <div class="meta">
                <strong>Data testu:</strong> {meta.get('test_date', 'N/A')}<br>
                <strong>ID Sesji:</strong> {meta.get('session_id', 'N/A')[:8]}...<br>
                <strong>Metoda:</strong> v{meta.get('method_version', '1.0.0')}
            </div>
    """
    
    # Validity
    status = validity.get("status", "unknown")
    status_map = {
        "valid": "Wiarygodny",
        "conditional": "Ważny z zastrzeżeniami",
        "invalid": "Nieważny"
    }
    status_label = status_map.get(status, status)
    
    html += f"""
            <div style="margin-bottom: 30px;">
                Status metodologiczny: <span class="badge {status}">{status_label}</span>
    """
    if status != "valid" and validity.get("issues"):
        html += '<ul style="margin-top: 10px; color: #e67e22;">'
        for issue in validity["issues"]:
            html += f"<li>{issue}</li>"
        html += "</ul>"
    html += "</div>"
    
    # Executive Summary (Interpretation)
    html += "<h2>Podsumowanie</h2>"
    overall_conf = interp.get("overall_confidence", 0)
    
    if overall_conf >= 0.7:
        summary_text = "Wyniki testu są spójne i stanowią solidną bazę do wyznaczenia stref treningowych."
    elif overall_conf >= 0.5:
        summary_text = "Wyniki są akceptowalne, choć wystąpiły pewne niespójności w sygnałach. Sugerowana ostrożność przy wyznaczaniu górnych stref."
    else:
        summary_text = "Ze względu na niską jakość danych lub konflikty sygnałów, wyniki należy traktować orientacyjnie."
        
    html += f'<p>{summary_text}</p>'
    
    # Thresholds
    html += "<h2>Wyznaczone Progi</h2>"
    
    # VT1
    vt1 = thresholds.get("vt1")
    if vt1:
        range_w = vt1["range_watts"]
        mid_w = vt1["midpoint_watts"]
        conf = vt1["confidence"]
        
        html += f"""
        <div class="card">
            <h3>Próg Aerobowy (VT1)</h3>
            <div class="range-display">{int(range_w[0])} – {int(range_w[1])} W</div>
            <div class="sub-text">Szacowany środek: ~{int(mid_w)} W</div>
            {_format_confidence(conf)}
            <p style="font-size: 0.9em; margin-top: 10px;">
                Górna granica strefy spokojnej wytrzymałości (Z2). Rozmowa staje się trudniejsza, ale wciąż możliwa.
            </p>
        </div>
        """
    
    # VT2
    vt2 = thresholds.get("vt2")
    if vt2:
        range_w = vt2["range_watts"]
        mid_w = vt2["midpoint_watts"]
        conf = vt2["confidence"]
        
        html += f"""
        <div class="card">
            <h3>Próg Beztlenowy (VT2)</h3>
            <div class="range-display">{int(range_w[0])} – {int(range_w[1])} W</div>
            <div class="sub-text">Szacowany środek: ~{int(mid_w)} W</div>
            {_format_confidence(conf)}
            <p style="font-size: 0.9em; margin-top: 10px;">
                Granica równowagi mleczanowej (FTP/CP). Powyżej tej intensywności zmęczenie narasta gwałtownie.
            </p>
        </div>
        """
        
    # SmO2 Context
    smo2 = data.get("smo2_context")
    if smo2:
        html += "<h2>Kontekst Mięśniowy (SmO₂)</h2>"
        html += """<div class="note">ℹ️ <strong>Sygnał Lokalny:</strong> Poniższe dane dotyczą tylko mięśnia pod sensorem i nie zastępują progów systemowych (VT).</div>"""
        
        deviation = smo2.get("deviation_from_vt1_watts")
        interp_smo2 = smo2.get("interpretation", "")
        
        if deviation is not None:
             html += f"""
             <p><strong>Relacja do VT1:</strong> {interp_smo2}</p>
             """
    
    # Conflicts
    if conflicts.get("detected"):
        html += "<h2>Uwagi i Konflikty</h2>"
        for c in conflicts["detected"]:
            severity = c.get("severity", "info")
            css_class = "conflict" if severity in ["warning", "critical"] else "note"
            
            html += f"""
            <div class="{css_class}">
                <strong>{c.get('description', '')}</strong><br>
                <span style="font-size: 0.9em;">{c.get('physiological_interpretation', '')}</span>
            </div>
            """
            
    # Recommendations (Interpretation)
    html += "<h2>Strefy Treningowe (Sugerowane)</h2>"
    
    qualifier = interp.get("confidence_level", "low")
    if qualifier == "low":
        html += """<div class="warning">⚠️ Ze względu na niską pewność wyników, poniższe strefy są jedynie przybliżeniem.</div>"""
        
    zones = interp.get("training_zones")
    if zones: # Assuming zones might be computed and put in interpretation later, or we construct them here roughly
        # For V1 spec, specific zones might not be in JSON explicitly unless added by build_result. 
        # But let's check if we have enough to show basic range advice based on VT1/VT2
        pass 
        
    if vt1 and vt2:
         vt1_mid = vt1["midpoint_watts"]
         vt2_mid = vt2["midpoint_watts"]
         
         html += f"""
         <table>
            <tr><th>Strefa</th><th>Zakres (W)</th><th>Cel</th></tr>
            <tr><td>Z1/Z2 (Wytrzymałość)</td><td>do {int(vt1['range_watts'][0])}</td><td>Baza tlenowa, regeneracja</td></tr>
            <tr><td>Z3 (Tempo)</td><td>{int(vt1['range_watts'][1])} – {int(vt2['range_watts'][0])}</td><td>Wytrzymałość siłowa, Sweet Spot</td></tr>
            <tr><td>Z4 (Próg)</td><td>{int(vt2['range_watts'][0])} – {int(vt2['range_watts'][1])}</td><td>Praca nad FTP</td></tr>
            <tr><td>Z5 (VO2max)</td><td>powyżej {int(vt2['range_watts'][1])}</td><td>Moc maksymalna tlenowa</td></tr>
         </table>
         """
    
    # Disclaimer
    html += """
        <div class="disclaimer">
            <p><strong>Nota prawna:</strong> Ten raport jest generowany automatycznie i służy wyłącznie celom informacyjnym. Nie stanowi porady medycznej. Przed rozpoczęciem intensywnego programu treningowego skonsultuj się z lekarzem.</p>
            <p>Generated by Ramp Test Methodology v1.0.0</p>
        </div>
        </div>
    </body>
    </html>
    """
    
    return html
