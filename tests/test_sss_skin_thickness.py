import pytest
import graphics.sss_skin_thickness as mod

CASES = [
    ("thickness_update", (1,)),
    ("lut_curvature", (1,)),
    ("kernel_rescale", (1,)),
    ("blur_order", (1,)),
    ("costume_reset", (1,)),
    ("energy_conserve", (1,)),
    ("profile", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("screen_blur", (1,)),
    ("metrics", (1,)),
    ("stretch", (1,)),
    ("transmit", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
