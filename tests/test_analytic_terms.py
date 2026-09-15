# import ps_theory_calculator
import numpy as np
import yaml, os
import pytest
import itertools as itt
from scipy.integrate import romb
from scipy.special import lpmv

from mentat_lss.emulator import ps_emulator
import mentat_lss._vendor.symbolic_pofk.linear as linear
from mentat_lss.models.analytic_terms import analytic_eft_model

def test_symbolic_pofk():
    k = np.linspace(0.01, 0.2, 25)
    test_plin = linear.plin_emulated(k, 0.8, 0.25, 0.05, 0.67, 0.96859)
    assert np.all(np.isinf(test_plin)) == False
    assert np.all(np.isnan(test_plin)) == False

@pytest.mark.parametrize("P_shot, a0, a2, kbins", [
    (0., 0., 0., np.linspace(0.01, 0.2, 25)),
    (1., 0., 0., np.linspace(0.01, 0.2, 25)),
    (0., 1., 0., np.linspace(0.01, 0.2, 25)),
    (0., 0., 1., np.linspace(0.01, 0.2, 25)),
    (1.5, 2., 0.5, np.linspace(0.01, 0.2, 25)),
])
def test_shotnoise_term_single_tracer(P_shot, a0, a2, kbins):
    num_zbins = 1; num_tracers = 1
    redshift_list = [0.1]*num_zbins
    ndens = np.random.rand(num_tracers, num_zbins)
    num_spectra = num_tracers * (num_tracers + 1) // 2

    model = analytic_eft_model(num_tracers, redshift_list, [0,2], kbins, ndens)
    params = np.array([0, 0, 0, P_shot, a0, a2])

    ps_anl = model.get_analytic_terms(params, [])
    assert ps_anl.shape == (num_spectra, num_zbins, len(kbins), 2)

    # set up the derived quantities by hand, since get_analytic_terms skips this
    # step entirely when all analytic parameters are zero
    model.set_params(params, [], model.get_required_analytic_parameters())
    model.calculate_pk_lin(model.k_lin, model.params)

    ps_expect = np.zeros((num_spectra, num_zbins, len(kbins), 2))
    # constant shot noise
    ps_expect[0, 0, :, 0] += P_shot / ndens[0, 0] / (model.params["alpha_perp"][0]**2 * model.params["alpha_para"][0])
    # k-dependent shot noise
    alpha_perp = model.params["alpha_perp"][0]
    alpha_para = model.params["alpha_para"][0]
    k_nl = model.get_k_nl(D=model.params["Dgrowth"][0])

    # (k, mu) coordinates the stochastic terms are actually evaluated at (AP effect)
    fac = np.sqrt(1 + model.mu**2 * ((alpha_perp / alpha_para)**2 - 1))
    mu_eval = model.mu * (alpha_perp / alpha_para) / fac
    k_eval = np.kron(kbins, fac).reshape(len(kbins), len(model.mu)) / alpha_perp

    pkmu = (k_eval / k_nl)**2 * (a0 * lpmv(0, 0, mu_eval) + a2 * lpmv(0, 2, mu_eval))
    pkmu /= ndens[0, 0] * (alpha_perp**2 * alpha_para)

    # project onto Legendre multipoles
    for i, ell in enumerate(model.ells):
        ps_expect[0, 0, :, i] += romb(pkmu * (2*ell + 1) * lpmv(0, ell, model.mu), dx=model.dmu, axis=1)

    assert np.allclose(ps_anl, ps_expect)

@pytest.mark.parametrize("counterterm_0, counterterm_2, counterterm_4, counterterm_fog, kbins", [
    (0., 0., 0., 0., np.linspace(0.01, 0.2, 25)),
    (10., 0., 0., 0., np.linspace(0.01, 0.2, 25)),
    (0., 11., 0., 0., np.linspace(0.01, 0.2, 25)),
    (0., 0., 12., 0., np.linspace(0.01, 0.2, 25)),
    (0., 0., 0., 13., np.linspace(0.01, 0.2, 25)),
    (14, 15, 16, 17, np.linspace(0.01, 0.2, 25)),
])
def test_counterterms_single_tracer(counterterm_0, counterterm_2, counterterm_4, counterterm_fog, kbins):
    num_zbins = 1; num_tracers = 1
    redshift_list = [0.1]*num_zbins
    ndens = np.random.rand(num_tracers, num_zbins)
    num_spectra = num_tracers * (num_tracers + 1) // 2

    ells = [0, 2, 4]
    model = analytic_eft_model(num_tracers, redshift_list, ells, kbins, ndens)
    params = np.array([counterterm_0, counterterm_2, counterterm_4, counterterm_fog, 0., 0., 0.])

    ps_anl = model.get_analytic_terms(params, [])
    assert ps_anl.shape == (num_spectra, num_zbins, len(kbins), len(ells))

    # set up the derived quantities by hand, since get_analytic_terms skips this
    # step entirely when all analytic parameters are zero
    model.set_params(params, [], model.get_required_analytic_parameters())
    model.calculate_pk_lin(model.k_lin, model.params)
    model.set_ir_resum_params(model.params["h"], 1.)

    alpha_perp = model.params["alpha_perp"][0]
    alpha_para = model.params["alpha_para"][0]
    f = model.params["fgrowth"][0]
    D = model.params["Dgrowth"][0]
    b1 = model.params["galaxy_bias_10_0_0"]

    # (k, mu) coordinates the counterterms are actually evaluated at (AP effect)
    fac = np.sqrt(1 + model.mu**2 * ((alpha_perp / alpha_para)**2 - 1))
    mu_eval = model.mu * (alpha_perp / alpha_para) / fac
    k_eval = np.kron(kbins, fac).reshape(len(kbins), len(model.mu)) / alpha_perp

    # IR-resummed linear power spectrum in redshift space, i.e. the no-wiggle part
    # plus a BAO-damped wiggle part
    plin = model.get_pk_lin(k_eval, D)
    plin_nw = model.irres.get_pk_nw(k_eval) * D**2
    Sigma2_tot = (1 + mu_eval**2 * f * (2 + f)) * model.Sigma2 + \
                 f**2 * mu_eval**2 * (mu_eval**2 - 1) * model.dSigma2
    plin_irres = plin_nw + np.exp(-k_eval**2 * D**2 * Sigma2_tot) * (plin - plin_nw)

    # leading and next-to-leading order counterterms
    ctr_LO = -2 * k_eval**2 * plin_irres * \
             (counterterm_0 + counterterm_2 * f * mu_eval**2 + counterterm_4 * f**2 * mu_eval**4)
    ctr_NLO = -1 * k_eval**4 * plin_irres * \
              counterterm_fog * f**4 * mu_eval**4 * (b1 + f * mu_eval**2)**2

    pkmu = (ctr_LO + ctr_NLO) / (alpha_perp**2 * alpha_para)

    # project onto Legendre multipoles
    ps_expect = np.zeros((num_spectra, num_zbins, len(kbins), len(ells)))
    for i, ell in enumerate(model.ells):
        ps_expect[0, 0, :, i] += romb(pkmu * (2*ell + 1) * lpmv(0, ell, model.mu), dx=model.dmu, axis=1)

    # the model evaluates the counterterms on a (256 x 51) (k, mu) grid and splines onto
    # the AP-mapped coordinates, while the above is exact, so the two agree only to the
    # accuracy of that interpolation (~3e-7 of the amplitude). Compare against the scale
    # of the spectrum rather than element-by-element, since the hexadecapole is a small
    # residual of a large cancellation and passes through zero.
    assert np.allclose(ps_anl, ps_expect, atol=1e-5 * np.abs(ps_expect).max())