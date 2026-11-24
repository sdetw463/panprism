import numpy as np
import pytest

from panprism.breakpoint_graph import VariantAllele, build_haplotype_graph_sequence


@pytest.fixture(scope="module")
def ref():
    rng = np.random.default_rng(22)
    return "".join(np.array(list("ACGT"))[rng.integers(0, 4, 300000)])


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("tss", [140000, 150000, 159999])
def test_inverted_surviving_tss_is_the_center(ref, strand, tss):
    v = VariantAllele(140000, ref[140000:160000], "<INV>", "inv", True, "INV_COMPLEX", 20000)
    m = build_haplotype_graph_sequence(ref, 0, tss, strand, [v], 131072)
    mapped = m.map_ref_interval_to_hap(tss, tss + 1)
    assert mapped[0][0] == m.tss_hap_idx == 65536


@pytest.mark.parametrize("strand", ["+", "-"])
def test_inverted_anchor_maps_reflected_offset_consistently(ref, strand):
    v = VariantAllele(140000, ref[140000:160000], "<INV>", "inv", True, "INV_COMPLEX", 20000)
    m = build_haplotype_graph_sequence(ref, 0, 147000, strand, [v], 131072)
    assert m.map_ref_interval_to_hap(147000, 147001)[0][0] == m.tss_hap_idx == 65536
    # Position 153000 is 6000 bp downstream on reference, reflected 6000 bp upstream inside the inversion
    mapped_other = m.map_ref_interval_to_hap(153000, 153001)[0]
    assert mapped_other[2] is True
    assert mapped_other[0] == 59536


@pytest.mark.parametrize("strand", ["+", "-"])
def test_indel_before_inversion_preserves_center_and_downstream_map(ref, strand):
    variants = [
        VariantAllele(130000, ref[130000], ref[130000] + "G" * 307, "ins", True, "INS_other", 307),
        VariantAllele(135000, ref[135000:135201], ref[135000], "del", True, "DEL", 200),
        VariantAllele(140000, ref[140000:160000], "<INV>", "inv", True, "INV_COMPLEX", 20000),
    ]
    m = build_haplotype_graph_sequence(ref, 0, 151000, strand, variants, 131072)
    assert m.map_ref_interval_to_hap(151000, 151001)[0][0] == m.tss_hap_idx == 65536
    a, b, inv = m.map_ref_interval_to_hap(165000, 165100)[0]
    assert not inv
    assert m.sequence[a:b] == ref[165000:165100]
