
import sys
import os

# Add project root to path
sys.path.append('/Users/wielkikrzych/Desktop/Tri_Dashboard')

from modules.calculations.interpretation import generate_training_advice

def test_interpretation():
    print("=== Test Interpretation & Prescription ===\n")
    
    # 1. Test "Aerobic Deficiency" Profile
    # VT1 (150) is very low compared to VT2 (300) -> Ratio 0.5
    metrics_deficiency = {
        'vt1_watts': 150,
        'vt2_watts': 300,
        'smo2_tau': 30
    }
    q_ok = {'is_valid': True}
    
    res_def = generate_training_advice(metrics_deficiency, q_ok)
    print(f"Deficiency Profile Diags: {res_def['diagnostics']}")
    print(f"Deficiency Profile Rx: {res_def['prescriptions']}")
    
    assert any("Aerobic Deficiency" in d for d in res_def['diagnostics']), "Should detect Aerobic Deficiency"
    assert any("Zone 2" in p for p in res_def['prescriptions']), "Should prescribe Zone 2"
    
    # 2. Test "Diesle" Profile
    # VT1 (260) close to VT2 (300) -> Ratio 0.86
    metrics_diesel = {
        'vt1_watts': 260,
        'vt2_watts': 300,
        'smo2_tau': 30
    }
    res_dies = generate_training_advice(metrics_diesel, q_ok)
    print(f"\nDiesel Profile Diags: {res_dies['diagnostics']}")
    
    assert any("Diesel Engine" in d for d in res_dies['diagnostics']), "Should detect Diesel profile"
    assert any("Polarized" in p for p in res_dies['prescriptions']), "Should prescribe Polarized/VO2max work"
    
    # 3. Test "Slow Recovery"
    metrics_slow = {
        'vt1_watts': 200,
        'vt2_watts': 300,
        'smo2_tau': 60 # Very Slow
    }
    res_slow = generate_training_advice(metrics_slow, q_ok)
    print(f"\nSlow Recovery Diags: {res_slow['diagnostics']}")
    
    assert any("Slow Recovery" in d for d in res_slow['diagnostics']), "Should detect Slow Recovery"
    
    # 4. Test Quality Gate
    q_bad = {'is_valid': False, 'issues': ['Bad Data']}
    res_bad = generate_training_advice(metrics_deficiency, q_bad)
    print(f"\nBad Quality Result Valid?: {res_bad['is_valid']}")
    
    assert res_bad['is_valid'] == False, "Should be invalid"
    assert not res_bad['diagnostics'], "Should have no diagnostics"
    
    print("\nInterpretation Verification Passed!")

if __name__ == "__main__":
    test_interpretation()
