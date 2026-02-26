
from modules.calculations.interpretation import generate_training_advice


def test_interpretation():
    """Test interpretation and prescription generation."""
    # 1. Test "Aerobic Deficiency" Profile
    # VT1 (150) is very low compared to VT2 (300) -> Ratio 0.5
    metrics_deficiency = {
        'vt1_watts': 150,
        'vt2_watts': 300,
        'smo2_tau': 30
    }
    q_ok = {'is_valid': True}
    
    res_def = generate_training_advice(metrics_deficiency, q_ok)
    
    assert any("Deficyt aerobowy" in d or "Aerobic Deficiency" in d for d in res_def['diagnostics']), "Should detect Aerobic Deficiency"
    assert any("Strefy 2" in p or "Zone 2" in p or "LSD" in p for p in res_def['prescriptions']), "Should prescribe Zone 2"
    
    # 2. Test "Diesel" Profile
    # VT1 (260) close to VT2 (300) -> Ratio 0.86
    metrics_diesel = {
        'vt1_watts': 260,
        'vt2_watts': 300,
        'smo2_tau': 30
    }
    res_dies = generate_training_advice(metrics_diesel, q_ok)
    
    assert any("Diesel" in d for d in res_dies['diagnostics']), "Should detect Diesel profile"
    assert any("spolaryzowany" in p.lower() or "polarized" in p.lower() for p in res_dies['prescriptions']), "Should prescribe Polarized/VO2max work"
    
    # 3. Test Quality Gate
    q_bad = {'is_valid': False, 'issues': ['Bad Data']}
    res_bad = generate_training_advice(metrics_deficiency, q_bad)
    
    assert res_bad['is_valid'] == False, "Should be invalid"
    assert not res_bad['diagnostics'], "Should have no diagnostics"
