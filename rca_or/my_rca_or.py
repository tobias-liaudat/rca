from __future__ import absolute_import, print_function
import numpy as np
from scipy.interpolate import Rbf
from modopt.signal.wavelet import get_mr_filters, filter_convolve
from modopt.opt.cost import costObj
from modopt.opt.proximity import Positivity
from modopt.opt.reweight import cwbReweight
from modopt.math.matrix import PowerMethod
import modopt.opt.algorithms as optimalg
import rca_or.proxs as rca_prox
import rca_or.grads as grads
import rca_or.utils as utils
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

def quickload(path):
    """ Load pre-fitted RCA model (saved with :func:`RCA.quicksave`).

    Parameters
    ----------
    path: str
        Path to where the fitted RCA model was saved.
    """
    if path[-4:] != '.npy':
        path += '.npy'
    RCA_params, fitted_model = np.load(path, allow_pickle=True)
    loaded_rca = RCA(**RCA_params)
    loaded_rca.obs_pos = fitted_model['obs_pos']
    loaded_rca.A = fitted_model['A']
    loaded_rca.S = fitted_model['S']
    loaded_rca.flux_ref = fitted_model['flux_ref']
    loaded_rca.psf_size = fitted_model['psf_size']
    loaded_rca.VT = fitted_model['VT']
    loaded_rca.alpha = fitted_model['alpha']
    loaded_rca.is_fitted = True
    return loaded_rca

class RCA(object):
    """ Resolved Components Analysis.

    Parameters
    ----------
    n_comp: int
        Number of components to learn.
    tobi_debug: bool
        Debug flag. If True opens up many options on the code and the workflow passes through the
        debugging blocks.
    upfact: int
        Upsampling factor. Default is 1 (no superresolution).
    ksig: float
        Value of :math:`k` for the thresholding in Starlet domain (taken to be
        :math:`k\sigma`, where :math:`\sigma` is the estimated noise standard deviation.)
    n_scales: int
        Number of Starlet scales to use for the sparsity constraint. Default is 3. Unused if
        ``filters`` are provided.
    ksig_init: float
        Similar to ``ksig``, for use when estimating shifts and noise levels, as it might
        be desirable to have it set higher than ``ksig``. Unused if ``shifts`` are provided
        when running :func:`RCA.fit`. Default is 5.
    filters: np.ndarray
        Optional filters to the transform domain wherein eigenPSFs are assumed to be sparse;
        convolution by them should amount to applying :math:`\Phi`. Optional; if not provided, the
        Starlet transform with `n_scales` scales will be used.
    verbose: bool or int
        If True, will only output RCA-specific lines to stdout. If verbose is set to 2,
        will run ModOpt's optimization algorithms in verbose mode.

    """
    def __init__(self, n_comp, tobi_debug = False, upfact=1, ksig=3, n_scales=3, ksig_init=5, filters=None,
                 verbose=2):
        self.n_comp = n_comp
        self.upfact = upfact
        self.ksig = ksig
        self.ksig_init = ksig_init

        # [TL] debugging
        self.tobi_debug = tobi_debug

        if filters is None:
            # option strings for mr_transform
            self.opt = ['-t2', '-n{}'.format(n_scales)]
            self.default_filters = True
        else:
            self.Phi_filters = filters
            self.default_filters = False
        self.verbose = verbose
        if self.verbose > 1:
            self.modopt_verb = True
        else:
            self.modopt_verb = False
        self.is_fitted = False

    def fit(self, obs_data, obs_pos, obs_weights=None, S=None, VT=None, alpha=None,
            hybrid_mode = 0, prox_option = 1, shifts=None, sigs=None, psf_size=None,
            psf_size_type='fwhm', flux=None, nb_iter=2, nb_subiter_S=200, nb_reweight=0,
            nb_subiter_weights=None, n_eigenvects=5, SNR_weights=None, graph_kwargs={}):
        """ Fits RCA to observed star field.

        Parameters
        ----------
        obs_data: np.ndarray
            Observed data.
        obs_pos: np.ndarray
            Corresponding positions.
        obs_weights: np.ndarray
            Corresponding weights. Can be either one per observed star, or contain pixel-wise values. Masks can be
            handled via binary weights. Default is None (in which case no weights are applied). Note if fluxes and
            shifts are not provided, weights will be ignored for their estimation. Noise level estimation only removes
            bad pixels (with weight strictly equal to 0) and otherwise ignores weights.
        S: np.ndarray
            First guess (or warm start) eigenPSFs :math:`S`. Default is ``None``.
        VT: np.ndarray
            Matrix of concatenated graph Laplacians. Default is ``None``.
        alpha: np.ndarray
            First guess (or warm start) weights :math:`\\alpha`, after factorization by ``VT``. Default is ``None``.
        hybrid_mode: int
            Integer that determines if the VT construction will use the graph constraint construction and the VT matrix
            passed as argument. Zero for only using the graph constraint. If different of zero there should be an alpha and
            VT matrix passed as argument. If 1, the optimization is done over the whole concatenated alpha matrix. If 2, after the
            alpha optimization, the zero position elements of the alpha matrix are maintained.
        prox_option: int
            Integer that the defines which proximal operator will be used on the alpha matrix optimization.
            prox_option = 0 => Fred's RCA sparse-strategy operator
            prox_option = 1 => l_1 proximal operator
            prox_option = 2 => l_2 proximal operator
            prox_option = 3 => l_21 proximal operator
            prox_option = 4 => ElasticNet proximal operator (l_2 + l_1)
        shifts: np.ndarray
            Corresponding sub-pixel shifts. Default is ``None``; will be estimated from
            observed data if not provided.
        sigs: np.ndarray
            Estimated noise levels. Default is ``None``; will be estimated from data
            if not provided.
        psf_size: float
            Approximate expected PSF size in pixels; will be used for the size of the Gaussian window for centroid estimation.
            ``psf_size_type`` determines the convention used for this size (default is FWHM).
            Ignored if ``shifts`` are provided. Default is Gaussian sigma of 7.5 pixels.
        psf_size_type: str
            Can be any of ``'R2'``, ``'fwhm'`` or ``'sigma'``, for the size defined from quadrupole moments, full width at half maximum
            (e.g. from SExtractor) or 1-sigma width of the best matching 2D Gaussian. Default is ``'fwhm'``.
        flux: np.ndarray
            Flux levels. Default is ``None``; will be estimated from data if not provided.
        nb_iter: int
            Number of overall iterations (i.e. of alternations). Note the weights do not
            get updated the last time around, so they actually get ``nb_iter-1`` updates.
            Default is 2.
        nb_subiter_S: int
            Maximum number of iterations for :math:`S` updates. If ModOpt's optimizers achieve
            internal convergence, that number may (and often is) not reached. Default is
            200.
        nb_reweight: int
            Number of reweightings to apply during :math:`S` updates. See equation (33) in RCA paper.
            Default is 0.
        nb_subiter_weights: int
            Maximum number of iterations for :math:`\\alpha` updates. If ModOpt's optimizers achieve
            internal convergence, that number may (and often is) not reached. Default is None;
            if not provided, will be set to ``2*nb_subiter_S`` (as it was in RCA v1).
        n_eigenvects: int
            Maximum number of eigenvectors to consider per :math:`(e,a)` couple. Default is ``None``;
            if not provided, *all* eigenvectors will be considered, which can lead to a poor
            selection of graphs, especially when data is undersampled. Ignored if ``VT`` and
            ``alpha`` are provided.
        SNR_weights: np array
            SNR weight for each stars.
        graph_kwargs: dictionary
            List of optional kwargs to be passed on to the :func:`utils.GraphBuilder`.
        """

        self.prox_option = prox_option
        self.hybrid_mode = hybrid_mode
        self.obs_data = np.copy(obs_data)
        self.shap = self.obs_data.shape
        self.im_hr_shape = (self.upfact*self.shap[0],self.upfact*self.shap[1],self.shap[2])
        self.obs_pos = obs_pos
        if obs_weights is None:
            self.obs_weights = np.ones(self.shap) #/ self.shap[2]
        elif obs_weights.shape == self.shap:
            self.obs_weights = obs_weights / np.expand_dims(np.sum(obs_weights,axis=2), 2) * self.shap[2]
        elif obs_weights.shape == (self.shap[2],):
            self.obs_weights = obs_weights.reshape(1,1,-1) / np.sum(obs_weights) * self.shap[2]
        else:
            raise ValueError(
            'Shape mismatch; weights should be of shape {} (for per-pixel weights) or {} (per-observation)'.format(
                             self.shap, self.shap[2:]))
        if S is None:
            if self.hybrid_mode == 0:
                self.S = np.zeros(self.im_hr_shape[:2] + (self.n_comp,))
            else:
                self.S = np.zeros(self.im_hr_shape[:2] + (self.n_comp + alpha.shape[0],))
        else:
            self.S = S
        self.VT = VT
        self.alpha = alpha
        self.shifts = shifts
        if shifts is None:
            self.psf_size = self._set_psf_size(psf_size, psf_size_type)
        self.sigs = sigs
        self.flux = flux
        self.nb_iter = nb_iter
        self.nb_subiter_S = nb_subiter_S
        if nb_subiter_weights is None:
            nb_subiter_weights = 2*nb_subiter_S
        self.nb_subiter_weights = nb_subiter_weights
        self.nb_reweight = nb_reweight
        self.n_eigenvects = n_eigenvects
        self.SNR_weights = SNR_weights
        self.graph_kwargs = graph_kwargs

        if self.verbose:
            print('Running basic initialization tasks...')
        self._initialize()
        if self.verbose:
            print('... Done.')
        if self.VT is None or self.alpha is None:
            if self.verbose:
                print('Constructing graph constraint...')
            self._initialize_graph_constraint()
            if self.verbose:
                print('... Done.')
        else:
            if self.hybrid_mode != 0:
                # Concatenate the alpha and the VT matrix
                user_def_alpha = np.copy(alpha)
                user_def_VT = np.copy(VT)
                self.VT = None
                self.alpha = None
                self.A = None

                # self.n_comp -= user_def_alpha.shape[0]
                if self.verbose:
                    print('Constructing graph constraint...')
                self._initialize_graph_constraint()
                if self.verbose:
                    print('... Done.')
                    print('Hybrid mode activated.')

                self.n_comp_hyb =  user_def_alpha.shape[0]
                self.n_vec_hyb = user_def_alpha.shape[1]
                self.n_vec_rca = self.VT.shape[0]

                print('alpha size: ' + str(self.alpha.shape))
                print('user_def_alpha size: ' + str(user_def_alpha.shape))
                print('VT size: ' + str(self.VT.shape))
                print('user_def_VT size: ' + str(user_def_VT.shape))

                zero_concat_1 = np.zeros((self.n_comp, self.n_vec_hyb))
                zero_concat_2 = np.zeros((self.n_comp_hyb,self.alpha.shape[1]))
                tmp_alpha_1 = np.concatenate((self.alpha,zero_concat_1), axis=1)
                tmp_alpha_2 = np.concatenate((zero_concat_2,user_def_alpha), axis=1)
                self.alpha = np.concatenate((tmp_alpha_1,tmp_alpha_2), axis=0)
                self.VT = np.concatenate((self.VT,user_def_VT), axis = 0)
                self.A = self.alpha.dot(self.VT)
                self.n_comp = self.alpha.shape[0]

                print('alpha size: ' + str(self.alpha.shape))
                print('VT size: ' + str(self.VT.shape))
                print('A size: ' + str(self.A.shape))
            elif self.hybrid_mode == 0:
                self.A = self.alpha.dot(self.VT)

        self._fit()
        self.is_fitted = True
        return self.S, self.A

    def quicksave(self, path):
        """ Save fitted RCA model for later use. Ideally, you would probably want to store the
        whole RCA instance, though this might mean storing a lot of data you are not likely to
        use if you do not alter the fit that was already performed.
        Stored models can be loaded with :func:`rca.quickload`.

        Parameters
        ----------
        path: str
            Path to where the fitted RCA model should be saved.
        """
        if not self.is_fitted:
            raise ValueError('RCA instance has not yet been fitted to observations. Please run\
            the fit method.')
        RCA_params = {'n_comp': self.n_comp, 'upfact': self.upfact}
        fitted_model = {'obs_pos': self.obs_pos, 'A': self.A, 'S': self.S,
                        'flux_ref': self.flux_ref, 'psf_size': self.psf_size,
                        'VT': self.VT, 'alpha': self.alpha}

        if path[-4:] != '.npy':
            path += '.npy'
        np.save(path, [RCA_params,fitted_model])

    def estimate_psf(self, test_pos, n_neighbors=15, rbf_function='thin_plate',
                     apply_degradation=False, shifts=None, flux=None,
                     upfact=None, rca_format=False):
        """ Estimate and return PSF at desired positions.

        Parameters
        ----------
        test_pos: np.ndarray
            Positions where the PSF should be estimated. Should be in the same format (units,
            etc.) as the ``obs_pos`` fed to :func:`RCA.fit`.
        n_neighbors: int
            Number of neighbors to use for RBF interpolation. Default is 15.
        rbf_function: str
            Type of RBF kernel to use. Default is ``'thin_plate'``.
        apply_degradation: bool
            Whether PSF model should be degraded (shifted and resampled on coarse grid),
            for instance for comparison with stars. If True, expects shifts to be provided.
            Default is False.
        shifts: np.ndarray
            Intra-pixel shifts to apply if ``apply_degradation`` is set to True.
        flux: np.ndarray
            Flux levels by which reconstructed PSF will be multiplied if provided. For comparison with
            stars if ``apply_degradation`` is set to True.
        upfact: int
            Upsampling factor; default is None, in which case that of the RCA instance will be used.
        rca_format: bool
            If True, returns the PSF model in "rca" format, i.e. with axises
            (n_pixels, n_pixels, n_stars). Otherwise, and by default, return them in
            "regular" format, (n_stars, n_pixels, n_pixels).
        """
        if not self.is_fitted:
            raise ValueError('RCA instance has not yet been fitted to observations. Please run\
            the fit method.')
        if upfact is None:
            upfact = self.upfact
        ntest = test_pos.shape[0]
        test_weights = np.empty((self.n_comp, ntest))
        for j,pos in enumerate(test_pos):
            # determine neighbors
            nbs, pos_nbs = utils.return_neighbors(pos, self.obs_pos, self.A.T, n_neighbors)
            # train RBF and interpolate for each component
            for i in range(self.n_comp):
                rbfi = Rbf(pos_nbs[:,0], pos_nbs[:,1], nbs[:,i], function=rbf_function)
                test_weights[i,j] = rbfi(pos[0], pos[1])
        PSFs = self._transform(test_weights)
        if apply_degradation:
            shift_kernels, _ = utils.shift_ker_stack(shifts,self.upfact)
            deg_PSFs = np.array([grads.degradation_op(PSFs[:,:,j], shift_kernels[:,:,j], upfact)
                                 for j in range(ntest)])
            if flux is not None:
                deg_PSFs *= flux.reshape(-1,1,1) / self.flux_ref
            if rca_format:
                return utils.rca_format(deg_PSFs)
            else:
                return deg_PSFs
        elif rca_format:
            return PSFs
        else:
            return utils.reg_format(PSFs)

    def validation_stars(self, test_stars, test_pos, response_flag = False):
        """ Match PSF model to stars - in flux, shift and pixel sampling - for validation tests.
        Returns both the matched PSFs' stamps and chi-square value.

        Parameters
        ----------
        test_stars: np.ndarray
            Star stamps to be used for comparison with the PSF model. Should be in "rca" format,
            i.e. with axises (n_pixels, n_pixels, n_stars).
        test_pos: np.ndarray
            Their corresponding positions.
        response_flag: bool
            Response option. True if in response mode.
        """
        if not self.is_fitted:
            raise ValueError('RCA instance has not yet been fitted to observations. Please run\
            the fit method.')
        if response_flag:
            test_shifts = np.zeros((test_pos.shape[0],2))
            test_fluxes = None
        else:
            cents = []
            for star in utils.reg_format(test_stars):
                cents += [utils.CentroidEstimator(star, sig=self.psf_size)]
            test_shifts = np.array([ce.return_shifts() for ce in cents])
            test_fluxes = utils.flux_estimate_stack(test_stars,rad=6) # [TL] changed 4 to 6

        matched_psfs = self.estimate_psf(test_pos, apply_degradation=True,
                                    shifts=test_shifts, flux=test_fluxes)
        return matched_psfs

    def _set_psf_size(self, psf_size, psf_size_type):
        """ Handles different "size" conventions."""
        if psf_size is not None:
            if psf_size_type == 'fwhm':
                return psf_size / (2*np.sqrt(2*np.log(2)))
            elif psf_size_type == 'R2':
                return np.sqrt(psf_size / 2)
            elif psf_size_type == 'sigma':
                return psf_size
            else:
                raise ValueError('psf_size_type should be one of "fwhm", "R2" or "sigma"')
        else:
            print('''WARNING: neither shifts nor an estimated PSF size were provided to RCA;
                    the shifts will be estimated from the data using the default Gaussian
                    window of 7.5 pixels.''')
            return 7.5

    def _initialize(self):
        """ Initialization tasks related to noise levels, shifts and flux. Note it includes
        renormalizing observed data, so needs to be ran even if all three are provided."""
        if self.default_filters:
            init_filters = get_mr_filters(self.shap[:2], opt=self.opt, coarse=True)
        else:
            init_filters = self.Phi_filters
        # noise levels
        if self.sigs is None:
            transf_data = utils.apply_transform(self.obs_data, init_filters)
            transf_mask = utils.transform_mask(self.obs_weights, init_filters[0])
            sigmads = np.array([1.4826*utils.mad(fs[0],w) for fs,w in zip(transf_data,
                                                      utils.reg_format(transf_mask))])
            self.sigs = sigmads / np.linalg.norm(init_filters[0])
        else:
            self.sigs = np.copy(self.sigs)
        self.sig_min = np.min(self.sigs)
        # intra-pixel shifts
        if self.shifts is None:
            thresh_data = np.copy(self.obs_data)
            cents = []
            for i in range(self.shap[2]):
                # don't allow thresholding to be over 80% of maximum observed pixel
                nsig_shifts = min(self.ksig_init, 0.8*self.obs_data[:,:,i].max()/self.sigs[i])
                thresh_data[:,:,i] = utils.HardThresholding(thresh_data[:,:,i], nsig_shifts*self.sigs[i])
                cents += [utils.CentroidEstimator(thresh_data[:,:,i], sig=self.psf_size)]
            self.shifts = np.array([ce.return_shifts() for ce in cents])
        self.shift_ker_stack,self.shift_ker_stack_adj = utils.shift_ker_stack(self.shifts,
                                                                              self.upfact)
        # flux levels
        if self.flux is None:
            #TODO: could actually pass on the centroids to flux estimator since we have them at this point
            self.flux = utils.flux_estimate_stack(self.obs_data,rad=6) # [TL] changed 4 to 6
        self.flux_ref = np.median(self.flux)
        # Normalize noise levels observed data
        self.sigs /= self.sig_min
        self.obs_data /= self.sigs.reshape(1,1,-1)

    def _initialize_graph_constraint(self):
        gber = utils.GraphBuilder(self.obs_data, self.obs_pos, self.obs_weights, self.n_comp,
                                  n_eigenvects=self.n_eigenvects, verbose=self.verbose,
                                  **self.graph_kwargs)
        self.VT, self.alpha, self.distances = gber.VT, gber.alpha, gber.distances
        self.sel_e, self.sel_a = gber.sel_e, gber.sel_a
        self.A = self.alpha.dot(self.VT)

    def _fit(self):
        weights = self.A
        comp = self.S
        alpha = self.alpha

        #### Source updates set-up ####
        # initialize dual variable and compute Starlet filters for Condat source updates
        dual_var = np.zeros((self.im_hr_shape))
        if self.default_filters:
            self.Phi_filters = get_mr_filters(self.im_hr_shape[:2], opt=self.opt, coarse=True, trim=False) # [JB] correction: Trim=Flase
        rho_phi = np.sqrt(np.sum(np.sum(np.abs(self.Phi_filters),axis=(1,2))**2))

        # Set up source updates, starting with the gradient
        source_grad = grads.SourceGrad(self.obs_data, self.obs_weights, weights, self.flux, self.sigs,
                                      self.shift_ker_stack, self.shift_ker_stack_adj,
                                      self.upfact, self.Phi_filters,self.SNR_weights)

        # sparsity in Starlet domain prox (this is actually assuming synthesis form)
        sparsity_prox = rca_prox.StarletThreshold(0) # we'll update to the actual thresholds later

        # and the linear recombination for the positivity constraint
        lin_recombine = rca_prox.LinRecombine(weights, self.Phi_filters)

        #### Weight updates set-up ####
        # gradient
        weight_grad = grads.CoeffGrad(self.obs_data, self.obs_weights, comp, self.VT, self.flux, self.sigs,
                                      self.shift_ker_stack, self.shift_ker_stack_adj, self.upfact,self.SNR_weights)

        # cost function
        weight_cost = costObj([weight_grad], verbose=self.modopt_verb)
        source_cost = costObj([source_grad], verbose=self.modopt_verb)

        # k-thresholding for spatial constraint
        if self.tobi_debug == True:
            # iter_func = lambda x: np.floor(x) + 1
            # coeff_prox = rca_prox.tobi_prox_l1(iter_func, self.nb_subiter_weights)
            # coeff_prox = rca_prox.tobi_prox_l1(iter_func, self.nb_iter)
            # coeff_prox.reset_iter()
            if self.prox_option == 0:
                iter_func = lambda x: np.floor(np.sqrt(x)) + 1
                coeff_prox = rca_prox.KThreshold(iter_func)
            elif self.prox_option == 1:
                iter_func = lambda x: np.floor(np.sqrt(x)) + 5
                coeff_prox = rca_prox.tobi_prox_l1(iter_func, self.nb_subiter_weights)
            elif self.prox_option == 2:
                 coeff_prox = rca_prox.tobi_prox_l2(prox_weights=None)
            elif self.prox_option == 3:
                 coeff_prox = rca_prox.tobi_prox_l21(prox_weights=None,iter_max=self.nb_subiter_weights)
            elif self.prox_option == 4:
                coeff_prox = rca_prox.tobi_prox_elasticNet(prox_weights=None,
                    gamma_param=0.1,iter_max=self.nb_subiter_weights)
            elif self.prox_option == 5:
                    coeff_prox = rca_prox.tobi_prox_l2_A(proxVT=self.VT)
            elif self.prox_option == 6:
                    coeff_prox = rca_prox.tobi_prox_unit_ball(proxVT=self.VT)
            elif self.prox_option == 7:
                    coeff_prox = rca_prox.tobi_prox_L2_and_ball(proxVT=self.VT)
        else:
            if self.prox_option == 0:
                iter_func = lambda x: np.floor(np.sqrt(x)) + 1
                coeff_prox = rca_prox.KThreshold(iter_func)
            elif self.prox_option == 1:
                iter_func = lambda x: np.floor(np.sqrt(x)) + 5
                coeff_prox = rca_prox.tobi_prox_l1(iter_func, self.nb_subiter_weights)
            elif self.prox_option == 2:
                 coeff_prox = rca_prox.tobi_prox_l2(prox_weights=None)
            elif self.prox_option == 3:
                 coeff_prox = rca_prox.tobi_prox_l21(prox_weights=None,iter_max=self.nb_subiter_weights)
            elif self.prox_option == 4:
                coeff_prox = rca_prox.tobi_prox_elasticNet(prox_weights=None,
                    gamma_param=0.1,iter_max=self.nb_subiter_weights)
            elif self.prox_option == 5:
                    coeff_prox = rca_prox.tobi_prox_l2_A(proxVT=self.VT)
            elif self.prox_option == 6:
                    coeff_prox = rca_prox.tobi_prox_unit_ball(proxVT=self.VT)
            elif self.prox_option == 7:
                    coeff_prox = rca_prox.tobi_prox_L2_and_ball(proxVT=self.VT)
            else:
                iter_func = lambda x: np.floor(np.sqrt(x))+1
                coeff_prox = rca_prox.KThreshold(iter_func)


        for k in range(self.nb_iter):
            #### Eigenpsf update ####
            # update gradient instance with new weights...
            # print('\nA before gradient update: \n\n')
            # print(weights)

            source_grad.update_A(weights)
            # print('\nA after gradient update: \n\n')
            # print(weights)
            # ... update linear recombination weights...
            lin_recombine.update_A(weights)
            # print('\nA after linear recombination: \n\n')
            # print(weights)
            # print('\nAA^T after linear recombination: \n\n')
            # print(weights.dot(weights.T))


            # ... set optimization parameters...
            beta = source_grad.spec_rad*np.max(self.SNR_weights) + rho_phi
            tau = 1./beta
            sigma = 1./lin_recombine.norm * beta/2

            # ... update sparsity prox thresholds...
            thresh = utils.reg_format(utils.acc_sig_maps(self.shap,self.shift_ker_stack_adj,self.sigs,
                                                        self.flux,self.flux_ref,self.upfact,weights,
                                                        sig_data=np.ones((self.shap[2],))*self.sig_min))
            thresholds = self.ksig*np.sqrt(np.array([filter_convolve(Sigma_k**2,self.Phi_filters**2)
                                              for Sigma_k in thresh]))

            sparsity_prox.update_threshold(tau*thresholds)

            # and run source update:
            transf_comp = utils.apply_transform(comp, self.Phi_filters)
            if self.nb_reweight:
                reweighter = cwbReweight(thresholds)
                for _ in range(self.nb_reweight):
                    source_optim = optimalg.Condat(transf_comp, dual_var, source_grad, sparsity_prox,
                                                   Positivity(), linear = lin_recombine, cost=source_cost,
                                                   max_iter=self.nb_subiter_S, tau=tau, sigma=sigma)
                    transf_comp = source_optim.x_final
                    reweighter.reweight(transf_comp)
                    thresholds = reweighter.weights
            else:
                source_optim = optimalg.Condat(transf_comp, dual_var, source_grad, sparsity_prox,
                                               Positivity(), linear = lin_recombine, cost=source_cost,
                                               max_iter=self.nb_subiter_S, tau=tau, sigma=sigma)
                transf_comp = source_optim.x_final
            comp = utils.rca_format(np.array([filter_convolve(transf_compj, self.Phi_filters, True)
                                    for transf_compj in transf_comp]))

            self.S = comp
            #TODO: replace line below with Fred's component selection (to be extracted from `low_rank_global_src_est_comb`)
            ind_select = range(comp.shape[2])

            #### Weight update ####
            if k < self.nb_iter-1:
                # [TL] debug
                if self.tobi_debug == True:
                    # update sources and reset iteration counter for K-thresholding
                    weight_grad.update_S(comp)
                    coeff_prox.reset_iter()

                    # initialize Power Method to compute spectral radius
                    # To calculate the beta (Lipschitz) parameter

                    # print('\n\n\n weight_grad.inv_spec_rad \n\n\n')
                    # print(weight_grad.inv_spec_rad)

                    # if abs(weight_grad.inv_spec_rad) < 0.001:
                    #     weight_grad.inv_spec_rad = 0.001

                    # print(np.dot(comp.T,comp).shape)
                    # spec_rad = np.max(np.linalg.eigvals(np.dot(comp.T,comp)))
                    # print('inv_spec_rad = %f \n weight_grad.inv_spec_rad = %f'
                    # %(1/spec_rad,weight_grad.inv_spec_rad)) # [TL]

                    # weight_optim = optimalg.ForwardBackward(alpha, weight_grad, coeff_prox, cost=weight_cost,
                    #                                 beta_param=1/spec_rad, auto_iterate=False)
                    # weight_grad.inv_spec_rad = 0.2
                    beta_param = 0.5/(weight_grad.spec_rad*np.max(self.SNR_weights))
                    weight_optim = optimalg.ForwardBackward(alpha, weight_grad, coeff_prox, cost=weight_cost,
                                                    beta_param=beta_param, auto_iterate=False)
                    # For the beta_param the inverse of the spectral radius of the matrix is used.
                    # This is the inverse of the maximium eigenvalue
                    weight_optim.iterate(max_iter=self.nb_subiter_weights)
                    alpha = weight_optim.x_final

                    if self.hybrid_mode == 2:
                        alpha[-self.n_comp_hyb:, :self.n_vec_rca] = 0
                        alpha[:self.n_comp - self.n_comp_hyb, -self.n_vec_hyb:] = 0

                    self.alpha = alpha

                    # if self.tobi_debug == True:
                    #     if self.n_eigenvects is None:
                    #         self.n_eigenvects = self.VT.shape[1]
                    #     np.save('/Users/tliaudat/Documents/PhD/codes/venv_p3/Data_preprocessing/rca_debug_proxl2/prox_L2/VT_it_%d.npy'%(k),self.VT)# [TL]
                    #     np.save('/Users/tliaudat/Documents/PhD/codes/venv_p3/Data_preprocessing/rca_debug_proxl2/prox_L2/alpha_it_%d.npy'%(k),self.alpha)# [TL]
                    #     np.save('/Users/tliaudat/Documents/PhD/codes/venv_p3/Data_preprocessing/rca_debug_proxl2/prox_L2/S_it_%d.npy'%(k),self.S)# [TL]
                    #     np.save('/Users/tliaudat/Documents/PhD/codes/venv_p3/Data_preprocessing/rca_debug_proxl2/prox_L2/A_it_%d.npy'%(k),self.A)# [TL]


                    weights_k = alpha.dot(self.VT)
                    # renormalize to break scale invariance
                    weight_norms = np.sqrt(np.sum(weights_k**2,axis=1))

                    # Printing stuff
                    # print('\n\n A line norms before normalization, it = %d\n'%(k))
                    # print(weight_norms)
                    # S_norm = np.zeros(weight_norms.shape)
                    # for S_it in range(comp.shape[2]):
                    #     S_norm[S_it] = np.sqrt(np.sum(comp[:,:,S_it]**2))
                    # print('\n\n S columns norms, it = %d\n'%(k))
                    # print(S_norm)

                    # THIS HAS NO SENSE [TL]
                    # comp *= weight_norms

                    #self.S = comp
                    weights_k /= weight_norms.reshape(-1,1)
                    #TODO: replace line below with Fred's component selection
                    ind_select = range(weights.shape[0])
                    weights = weights_k[ind_select,:]
                    supports = None #TODO



                    # Plot stuff
                    plot_option = False
                    if k%1 == 0 and plot_option == True:
                        my_path = '/Users/tliaudat/Documents/PhD/codes/venv_p3/Data_preprocessing/rca_debug_proxl2/prox_L2/11_50_100elastic/'
                        my_test_nb = 1
                        top = mpl.cm.get_cmap('Oranges_r', 128)
                        bottom = mpl.cm.get_cmap('Blues', 128)
                        newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                                               bottom(np.linspace(0, 1, 128))))
                        newcmp = ListedColormap(newcolors, name='OrangeBlue')
                        mpl.use('agg')
                        if k == 0:
                            vmax1_alpha = np.max([abs(np.max(alpha)), abs(np.min(alpha))])
                            vmax1_A = np.max([abs(np.max(weights)), abs(np.min(weights))])
                        fig, axs = plt.subplots(1, 2, figsize=(8,8))

                        im_1 = axs[0].imshow(weights.T, cmap=newcmp, vmin=-vmax1_A, vmax=vmax1_A)
                        axs[0].set_title('A.T it=%d'%(k))
                        cbar = fig.colorbar(im_1, ax=axs[0], extend='both')
                        cbar.minorticks_on()

                        im_2 = axs[1].imshow(alpha.T, cmap=newcmp, vmin=-vmax1_alpha, vmax=vmax1_alpha)
                        axs[1].set_title('alpha.T it=%d'%(k))
                        cbar = fig.colorbar(im_2, ax=axs[1], extend='both')
                        cbar.minorticks_on()

                        plt.savefig(my_path + 'A_alpha_test%d-it_%d.png'%(my_test_nb,k))
                        plt.close()


                else:
                    # update sources and reset iteration counter for K-thresholding
                    weight_grad.update_S(comp)
                    coeff_prox.reset_iter()

                    beta_param = 0.5/(weight_grad.spec_rad*np.max(self.SNR_weights))
                    weight_optim = optimalg.ForwardBackward(alpha, weight_grad, coeff_prox, cost=weight_cost,
                                                    beta_param=beta_param, auto_iterate=False)
                    weight_optim.iterate(max_iter=self.nb_subiter_weights)
                    alpha = weight_optim.x_final
                    self.alpha = alpha

                    if self.hybrid_mode == 2:
                        alpha[-self.n_comp_hyb:, :self.n_vec_rca] = 0
                        alpha[:self.n_comp - self.n_comp_hyb, -self.n_vec_hyb:] = 0

                    weights_k = alpha.dot(self.VT)
                    # renormalize to break scale invariance
                    weight_norms = np.sqrt(np.sum(weights_k**2,axis=1))
                    # comp *= weight_norms # [TL] BUG
                    weights_k /= weight_norms.reshape(-1,1)
                    #TODO: replace line below with Fred's component selection
                    ind_select = range(weights.shape[0])
                    weights = weights_k[ind_select,:]
                    supports = None #TODO


        self.A = weights
        self.S = comp
        self.alpha = alpha
        source_grad.MX(transf_comp)
        self.current_rec = source_grad._current_rec

    def _transform(self, A):
        return self.S.dot(A)
