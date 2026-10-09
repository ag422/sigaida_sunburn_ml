import numpy as np
import pytest

from sunrisk import config, med, sunscreen, uv


def test_uvi_irradiance_roundtrip():
    assert uv.uvi_to_irradiance(8) == pytest.approx(0.2)
    assert uv.irradiance_to_uvi(0.2) == pytest.approx(8)


def test_integrate_dose_constant():
    t = np.arange(0, 601, 60.0)
    assert uv.integrate_dose(t, np.full(t.size, 0.2)) == pytest.approx(120.0)


def test_prior_median_by_type():
    assert med.prior_median(2) == 250.0
    with pytest.raises(ValueError):
        med.prior_median(7)


def test_burn_probability_is_half_at_med():
    assert med.burn_probability(300, 300) == pytest.approx(0.5)
    assert med.burn_probability(600, 300) > 0.95


def test_personal_med_moves_toward_evidence():
    m = med.PersonalMED(3)
    before = m.median()
    for _ in range(3):
        m.update(200.0, burned=True)  # burned well below the type-III prior median
    assert m.median() < before
    m2 = med.PersonalMED(3)
    m2.update(500.0, burned=False)
    assert m2.median() > before


def test_soft_label_is_between_hard_labels():
    hard_yes, soft, hard_no = (med.PersonalMED(2) for _ in range(3))
    hard_yes.update(250, True); soft.update(250, 0.5); hard_no.update(250, False)
    assert hard_yes.median() < soft.median() < hard_no.median()


def test_med_samples_match_quantiles():
    m = med.PersonalMED(2)
    x = m.sample(np.random.default_rng(1), 20000)
    assert np.quantile(x, 0.1) == pytest.approx(m.quantile(0.1), rel=0.03)


def test_spf_linear_in_amount():
    # Faurschou 2007 linear model: half the lab amount -> (SPF-1)/2 + 1
    assert sunscreen.effective_spf(30, 2.0) == pytest.approx(30)
    assert sunscreen.effective_spf(30, 1.0) == pytest.approx(15.5)
    assert sunscreen.effective_spf(30, 0.8) == pytest.approx(1 + 29 * 0.4)


def test_sunscreen_wear_and_swim():
    s = sunscreen.SunscreenState()
    s.apply(30, 2.0)
    s.wear(config.SUNSCREEN_WEAR_TAU_H.value)
    assert s.effective_spf() == pytest.approx(1 + 29 * np.exp(-1))
    before = s.effective_spf() - 1
    s.swim()
    assert s.effective_spf() - 1 == pytest.approx(before * config.SWIM_RETENTION.value)
