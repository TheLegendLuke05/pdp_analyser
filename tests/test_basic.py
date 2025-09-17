import pdp_analyser

def test_imports():
    assert hasattr(pdp_analyser, "variable_extractor")
    assert hasattr(pdp_analyser, "overall_plot")
