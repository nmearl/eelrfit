import pytest

from eelrfit.lines import get_line, line_from_cloudy_label, species_from_label


def test_line_names_and_labels():
    assert get_line("Hb") is get_line("HI_4861") is get_line("hbeta")
    assert get_line("OIII_5007").label == "[OIII] 5007"
    assert get_line("OIII_5007").cloudy_label == "O  3 5006.84A"
    assert line_from_cloudy_label("O  3 5006.84A").id == "OIII_5007"
    assert line_from_cloudy_label("Blnd 5875.66A").id == "HeI_5876"
    assert line_from_cloudy_label("Fe 2 5158.00A") is None
    assert species_from_label("[SII]6717") == "SII" and species_from_label("Halpha") == "HI"
    assert species_from_label("He") is None  # pyPipe3D's H-epsilon label is ambiguous
    with pytest.raises(KeyError, match="Unknown emission line"):
        get_line("XX_1234")
