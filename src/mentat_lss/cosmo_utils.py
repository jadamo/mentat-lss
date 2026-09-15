import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import quad, solve_ivp
from scipy.special import spherical_jn
from scipy.fft import dst, idst

# NOTE: code as-is takes ~0.1 ms to calculate fgrowth in this method
class LCDMCosmology():

    def __init__(self, h, Omega_m0, Omega_K0=0.):
        self.params = {}
        self.params['h'] = h
        self.params['Omega_m0'] = Omega_m0
        self.params['Omega_K0'] = Omega_K0
        # self.params['H_0'] = self.params['h'] * 100
        self.params['H_0_in_Mpc_inv'] = self.params['h'] * 100 / (299792458 * 1e-3) # in unit of 1/Mpc
        self.params['omega_m0'] = self.params['Omega_m0'] * self.params['h']**2
        self.params['Omega_de0'] = 1 - self.params['Omega_m0'] - self.params['Omega_K0']

    @staticmethod
    def get_a(redshift):
        return 1. / (1 + redshift)

    def get_E(self, redshift):
        Omega_m0 = self.params['Omega_m0']
        Omega_de0 = self.params['Omega_de0']
        Omega_K0 = self.params['Omega_K0']
        E = np.sqrt(Omega_m0 * (1 + redshift)**3 + Omega_K0 * (1 + redshift)**2 + Omega_de0)
        return E

    def get_Omega_m(self, redshift):
        return self.params['Omega_m0'] / self.get_E(redshift)**2 * (1 + redshift)**3

    def get_Omega_de(self, redshift):
        return self.params['Omega_de0'] / self.get_E(redshift)**2

    def get_Omega_K(self, redshift):
        return 1 - self.get_Omega_m(redshift) - self.get_Omega_de(redshift)

    # linear growth factor in LCDM cosmology
    def get_Dgrowth(self, redshift):
        a = 1. / (1 + redshift)
        Omega_m = self.get_Omega_m(redshift)
        Omega_de = self.get_Omega_de(redshift)
        Omega_K = self.get_Omega_K(redshift)
        res = quad(lambda t: (Omega_m / t + Omega_de * t**2 + Omega_K)**(-3./2), 0., 1.)[0]
        Dgrowth = (5./2.) * a * Omega_m * res
        return Dgrowth

    # linear growth rate in LCDM cosmology
    def get_fgrowth(self, redshift):
        Omega_m = self.get_Omega_m(redshift)
        Omega_de = self.get_Omega_de(redshift)
        Omega_K = self.get_Omega_K(redshift)
        res = quad(lambda t: (Omega_m / t + Omega_de * t**2 + Omega_K)**(-3./2), 0., 1.)[0]
        fgrowth = -1 - Omega_m / 2 + Omega_de + 1 / res
        return fgrowth

    # in comoving Mpc
    def get_comoving_dist(self, redshift):
        comoving_dist = quad(lambda t: 1. / self.get_hubble_in_Mpc_inv(t), 0., redshift)[0]
        return comoving_dist

    def get_hubble_in_Mpc_inv(self, redshift):
        return self.params['H_0_in_Mpc_inv'] * self.get_E(redshift)

    @staticmethod
    def comoving_to_radial(x, K):
        if K == 0: return x
        elif K > 0: return np.sin(np.sqrt(K) * x) / np.sqrt(K)
        elif K < 0: return np.sinh(np.sqrt(-K) * x) / np.sqrt(-K)
        else: raise ValueError('The curvature K is not specified.')

    # in comoving Mpc
    def get_D_angular(self, redshift, K=0):
        return self.comoving_to_radial(self.get_comoving_dist(redshift), K)

    # in comoving Mpc/h
    def get_D_angular_in_h_inv_Mpc(self, redshift, K=0):
        return self.get_D_angular(redshift, K) * self.params['h']

    def set_Omega_m0(self, Omega_m0):
        self.params["Omega_m0"] = Omega_m0
        self.params['omega_m0'] = self.params['Omega_m0'] * self.params['h']**2
        self.params['Omega_de0'] = 1 - self.params['Omega_m0'] - self.params['Omega_K0']

    def set_h(self, h):
        self.params["h"] = h
        self.params['omega_m0'] = self.params['Omega_m0'] * self.params['h']**2
        self.params['Omega_de0'] = 1 - self.params['Omega_m0'] - self.params['Omega_K0']
        self.params['H_0_in_Mpc_inv'] = self.params['h'] * 100 / (299792458 * 1e-3) # in unit of 1/Mpc

class w0waCosmology(LCDMCosmology):
    """For derivation, see https://arxiv.org/pdf/astro-ph/0208512"""
    
    def __init__(self, h, Omega_m0, w0, wa, Omega_K0=0.):
        super().__init__(h, Omega_m0, Omega_K0)
        self.params['w0'] = w0
        self.params['wa'] = wa
        self._growth_sol = None 

    def _get_f_de(self, redshift):
        """Dark energy density scaling: rho_de(a)/rho_de0 for CPL w(a)=w0+wa*(1-a)."""
        a = self.get_a(redshift)
        w0 = self.params['w0']
        wa = self.params['wa']
        return a**(-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - a))

    def get_E(self, redshift):
        Omega_m0 = self.params['Omega_m0']
        Omega_de0 = self.params['Omega_de0']
        Omega_K0 = self.params['Omega_K0']
        f_de = self._get_f_de(redshift)
        return np.sqrt(Omega_m0 * (1 + redshift)**3 + Omega_K0 * (1 + redshift)**2 + Omega_de0 * f_de)

    def get_Omega_de(self, redshift):
        return self.params['Omega_de0'] * self._get_f_de(redshift) / self.get_E(redshift)**2

    def _get_dlnE_dlna(self, a):
        """d ln E / d ln a from the Friedmann + acceleration equations.
        Exact for any w(a) (unlike older growing-mode integral)."""
        redshift = 1. / a - 1.
        Omega_m = self.get_Omega_m(redshift)
        Omega_de = self.get_Omega_de(redshift)
        Omega_K = self.get_Omega_K(redshift)
        w_a = self.params['w0'] + self.params['wa'] * (1 - a)
        return -(3. / 2.) * Omega_m - Omega_K - (3. / 2.) * (1 + w_a) * Omega_de

    def _solve_growth(self, a_min=1e-3):
        """Integrate the linear growth ODE D'' + (2 + dlnE/dlna) D' - (3/2) Omega_m(a) D = 0
        (' = d/dlna) from deep matter domination (D=a) to a=1.

        The Heath/Peebles closed-form integral used in LCDMCosmology.get_Dgrowth
        is exact only when the non-matter component behaves like a cosmological
        constant or curvature. For general w0-wa dark energy it does not solve
        this equation, so the growth factor must be integrated numerically.
        """
        def rhs(lna, y):
            a = np.exp(lna)
            D, Dp = y
            Omega_m = self.get_Omega_m(1. / a - 1.)
            return [Dp, -(2 + self._get_dlnE_dlna(a)) * Dp + 1.5 * Omega_m * D]
        
        y0 = [a_min, a_min]  # matter-dominated growing mode: D=a, dD/dlna=a
        self._growth_sol = solve_ivp(rhs, [np.log(a_min), 0.], y0,
                                      dense_output=True, rtol=1e-10, atol=1e-12)

    # linear growth factor for w0wa dark energy (unnormalized; callers use ratios)
    def get_Dgrowth(self, redshift):
        if self._growth_sol is None:
            self._solve_growth()
        D, _ = self._growth_sol.sol(np.log(self.get_a(redshift)))
        return D

    # linear growth rate for w0wa dark energy
    def get_fgrowth(self, redshift):
        if self._growth_sol is None:
            self._solve_growth()
        D, Dp = self._growth_sol.sol(np.log(self.get_a(redshift)))
        return Dp / D


class IRResum:

    def __init__(self, pk_lin, hubble, rbao=110, khmin=7e-5, khmax=7, n_min=120, n_max=240,
                kmin_interp=1e-7, kmax_interp=1e+7, kwarg={}):
        self.rbao = rbao
        kh = np.linspace(khmin, khmax, 2**16) # in unit of 1/Mpc
        plin = pk_lin(kh / hubble, **kwarg) * hubble**(-3) # in unit of Mpc^3
        plin_nw = self.remove_wiggle(kh, plin, n_min, n_max) # in unit of Mpc^3
        
        # ad-hoc adjustment at high k for extrapolation
        plin_nw[-10:] = plin[-10:]

        # extrapolation
        k_low = np.geomspace(kmin_interp, kh[0] / hubble, 100)[:-1]
        k_high = np.geomspace(kh[-1] / hubble, kmax_interp, 100)[1:]
        k_extrap = np.hstack((k_low, kh / hubble, k_high)) # in unit of h/Mpc
        plin_nw_extrap = np.hstack((pk_lin(k_low, **kwarg), plin_nw * hubble**3, pk_lin(k_high, **kwarg))) # in unit of (Mpc/h)^3

        # spline interpolation
        self.pk_nw_interp = ius(np.log(k_extrap), np.log(plin_nw_extrap))

    def remove_wiggle(self, kh, plin, n_min, n_max):
        # wiggly-non-wiggly splitting of linear power spectrum using DST (Sec. 4.2 of arXiv:2004.10607)

        harms = dst(np.log(kh * plin))

        n = np.arange(1,len(harms)+1)
        i_odd = np.arange(0,len(harms)-1,2)
        i_even = np.arange(1,len(harms),2)

        n_odd = n[i_odd]
        n_even = n[i_even]
        harms_odd = harms[i_odd]
        harms_even = harms[i_even]

        n = n[:int(len(harms)/2)]
        n_sd = np.hstack((n[n <= n_min], n[n >= n_max]))
        harms_odd_sd = np.hstack((harms_odd[n <= n_min], harms_odd[n >= n_max]))
        harms_even_sd = np.hstack((harms_even[n <= n_min], harms_even[n >= n_max]))

        harms_odd_s = ius(n_sd, harms_odd_sd)(n)
        harms_even_s = ius(n_sd, harms_even_sd)(n)

        i_rec = np.argsort(np.hstack((n_odd, n_even)))
        harms_s = np.hstack((harms_odd_s, harms_even_s))[i_rec]
        plin_nw = np.exp(idst(harms_s)) / kh # in unit of Mpc^3

        return plin_nw

    def get_pk_nw(self, k):
        return np.exp(self.pk_nw_interp(np.log(k)))

    def get_Sigma2(self, ks, kmin=1e-7, limit=1000):
        res = quad(lambda q: self.get_pk_nw(q) * (1 - spherical_jn(0,self.rbao*q) + 2 * spherical_jn(2,self.rbao*q)), kmin, ks, limit=limit, epsrel=1e-6)
        return res[0] / (6*np.pi**2)

    def get_dSigma2(self, ks, kmin=1e-7, limit=1000):
        res = quad(lambda q: self.get_pk_nw(q) * spherical_jn(2,self.rbao*q), kmin, ks, limit=limit, epsrel=1e-6)
        return res[0] / (2*np.pi**2)

    def get_sigmav2(self, kmin=1e-7, kmax=1e+7, limit=1000):
        res = quad(lambda q: self.get_pk_nw(q), kmin, kmax, limit=limit, epsrel=1e-6)
        return res[0] / (6*np.pi**2)

    def get_Sigma2_rsd(self, fgrowth, mu, ks=0.2):
        Sigma2_1 = (1 + mu**2 * fgrowth * (2 + fgrowth)) * self.get_Sigma2(ks)
        Sigma2_2 = fgrowth**2 * mu**2 * (mu**2 - 1) * self.get_dSigma2(ks)
        return Sigma2_1 + Sigma2_2
    
def get_log_extrap(x, y, xmin, xmax):
    x_low = x_high = []
    y_low = y_high = []
    
    if xmin < x[0]:
        dlnx_low = np.log(x[1]/x[0])
        num_low = int(np.log(x[0]/xmin) / dlnx_low) + 1
        x_low = x[0] * np.exp(dlnx_low * np.arange(-num_low, 0))
        
        dlny_low = np.log(y[1]/y[0])
        y_low = y[0] * np.exp(dlny_low * np.arange(-num_low, 0))

    if xmax > x[-1]:
        dlnx_high= np.log(x[-1]/x[-2])
        num_high = int(np.log(xmax/x[-1]) / dlnx_high) + 1
        x_high = x[-1] * np.exp(dlnx_high * np.arange(1, num_high+1))

        dlny_high = np.log(y[-1]/y[-2])
        y_high = y[-1] * np.exp(dlny_high * np.arange(1, num_high+1))

    x_extrap = np.hstack((x_low, x, x_high))
    y_extrap = np.hstack((y_low, y, y_high))
    return x_extrap, y_extrap