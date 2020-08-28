""" Defines proximal operators to be fed to ModOpt algorithm that are specific to RCA
(or rather, not currently in ``modopt.opt.proximity``)."""

from __future__ import absolute_import, print_function
import numpy as np
from modopt.signal.noise import thresh
from modopt.opt.linear import LinearParent
from modopt.signal.wavelet import filter_convolve
from rca_or.utils import lineskthresholding, reg_format, rca_format, SoftThresholding, apply_transform, op_tobi_prox_l1

class LinRecombine(object):
    """ Multiply eigenvectors and (factorized) weights."""
    def __init__(self, A, filters, compute_norm=False):
        self.A = A
        self.op = self.recombine
        self.adj_op = self.adj_rec
        self.filters = filters
        if compute_norm:
            U, s, Vt = np.linalg.svd(self.A.dot(self.A.T),full_matrices=False)
            self.norm = np.sqrt(s[0])

    def recombine(self, transf_S):
        S = np.array([filter_convolve(transf_Sj, self.filters, filter_rot=True)
                      for transf_Sj in transf_S])
        return rca_format(S).dot(self.A)

    def adj_rec(self, Y):
        return apply_transform(Y.dot(self.A.T), self.filters)

    def update_A(self, new_A, update_norm=True):
        self.A = new_A
        if update_norm:
            # print('norm after = %f\n'%(self.norm)) # [TL]
            U, s, Vt = np.linalg.svd(self.A.dot(self.A.T),full_matrices=False)
            self.norm = np.sqrt(s[0])
            # print('norm after = %f\n'%(self.norm))

class KThreshold(object):
    """This class defines linewise hard-thresholding operator with variable thresholds.

    Parameters
    ----------
    iter_func: function
        Input function that calcultates the number of non-zero values to keep in each line at each iteration.
    """
    def __init__(self, iter_func):

        self.iter_func = iter_func
        self.iter = 0

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0


    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        """
        self.iter += 1
        return lineskthresholding(data,self.iter_func(self.iter))

    def cost(self, x):
        """Returns 0 (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0

class tobi_prox_l1(object):
    """This class defines the classic l1 prox with GMCA-like decreasing weighting values.

    Parameters
    ----------
    iter_func: function
        Input function that calcultates the number of non-zero values to keep in each line at each iteration.
    """
    def __init__(self, iter_func, kmax):

        self.iter_func = iter_func
        self.iter = 0
        self.iter_max = kmax

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0


    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        """
        self.iter += 1
        return op_tobi_prox_l1(data,self.iter,self.iter_max)

    def cost(self, x):
        """Returns 0 (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0

class tobi_prox_l2(object):
    """This class defines the classic l2 prox.
    « Mixed-norm estimates for the M/EEG inverse problem using accelerated gradient methods
Alexandre Gramfort, Matthieu Kowalski, Matti Hämäläinen »

    Parameters
    ----------
    prox_weights: Matrix
        Corresponds to the weights of the weighted norm l_{w,2}. They are set by default to ones.
    beta_param: float number
        Corresponds to the beta (or lambda) parameter that goes with the fucn tion we will
        calculate the prox on. prox_{lambda f(.)}(y)
    iter: Integer
        Iteration number, just to follow track of the iterations. It could be part of the lambda
        update strategy for the prox calculation.
    """
    def __init__(self, prox_weights = None):
        self.prox_weights = prox_weights
        self.beta_param = 0
        self.iter = 0

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        The extra factor is the beta_param!!
        Should be used on the proximal operator function
        """
        self.beta_param = extra_factor
        # print('Using L2 prox.. beta = %.5f it=%d\n'%(self.beta_param,self.iter))

        if self.prox_weights is None:
            self.prox_weights = np.ones(data.shape)

        self.iter += 1 # not used in this prox

        result = self.op_tobi_prox_l2(data)
        # print('Prox results, it = %d'%(self.iter))
        # print(result)
        return result

    def op_tobi_prox_l2(self, data):
        """ Apply the opterator on the whole data matrix
        for a vector: x = prox_{lambda || . ||^{2}_{w,2}}(y) =>
        x_i = y_i /(1 + lambda w_i)
        The operator can be used for the whole data matrix at once.
        """
        dividing_weigts =  np.ones(data.shape) + self.beta_param * self.prox_weights

        return np.copy(data/dividing_weigts)

    def cost(self, x):
        """Returns 0 (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0

class tobi_prox_l2_A(object):
    """This class defines the l2 prox on -> f(alpha) = beta ||alpha VT||_F^2
    Theorem 6.15 from: SIAM, Chapter 6, The Proximal Operator.
    First-Order Methods in Optimization, Amir Beck

    Parameters
    ----------
    proxVT: Matrix
        Corresponds to the VT matrix from RCA.
    beta_param: float number
        Corresponds to the beta (or lambda) parameter that goes with the fucn tion we will
        calculate the prox on. prox_{lambda f(.)}(y)
    iter: Integer
        Iteration number, just to follow track of the iterations. It could be part of the lambda
        update strategy for the prox calculation.
    """
    def __init__(self, proxVT):
        self.proxVT = proxVT
        self.beta_param = 0
        self.iter = 0

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        The extra factor is the beta_param!!
        Should be used on the proximal operator function
        """
        self.beta_param = extra_factor
        # print('Using L2 prox.. beta = %.5f it=%d\n'%(self.beta_param,self.iter))

        self.iter += 1 # not used in this prox
        result = self.op_tobi_prox_l2_A(data)
        # result = self.op_tobi_prox_l2_unit_ball(data)
        # print('Prox results, it = %d'%(self.iter))
        # print(result)
        return result

    def op_tobi_prox_l2_A(self, data):
        """ Apply the opterator on the whole data matrix
        for a vector: x = prox_{lambda || . ||^{2}_{w,2}}(y) =>
        x_i = y_i /(1 + lambda w_i)
        The operator can be used for the whole data matrix at once.
        """
        lambda_param = np.mean(np.diag(self.proxVT.dot(self.proxVT.T)))
        alphaVT = data.dot(self.proxVT)
        div_w = np.ones(alphaVT.shape) + lambda_param*self.beta_param

        return data + (1/lambda_param)*(alphaVT/(div_w) - alphaVT).dot(self.proxVT.T)

    def cost(self, x):
        """Returns 0 (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0


class tobi_prox_unit_ball(object):
    """This class defines the l2 prox on -> f(alpha) = beta ||alpha VT||_F^2
    Theorem 6.15 from: SIAM, Chapter 6, The Proximal Operator.
    First-Order Methods in Optimization, Amir Beck

    Parameters
    ----------
    proxVT: Matrix
        Corresponds to the VT matrix from RCA.
    beta_param: float number
        Corresponds to the beta (or lambda) parameter that goes with the fucn tion we will
        calculate the prox on. prox_{lambda f(.)}(y)
    iter: Integer
        Iteration number, just to follow track of the iterations. It could be part of the lambda
        update strategy for the prox calculation.
    """
    def __init__(self, proxVT):
        self.proxVT = proxVT
        self.beta_param = 0
        self.iter = 0

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        The extra factor is the beta_param!!
        Should be used on the proximal operator function
        """
        self.beta_param = extra_factor
        # print('Using L2 prox.. beta = %.5f it=%d\n'%(self.beta_param,self.iter))

        self.iter += 1 # not used in this prox
        # result = self.op_tobi_prox_l2_A(data)
        result = self.op_tobi_prox_l2_unit_ball(data)
        # print('Prox results, it = %d'%(self.iter))
        # print(result)
        return result

    def op_tobi_prox_l2_unit_ball(self, data):
        """ Apply the opterator unit ball constraint.
        The operator can be used for the whole data matrix at once.
        """
        alphaVT = data.dot(self.proxVT)
        weight_norms = np.sqrt(np.sum(alphaVT**2,axis=1))
        data /= weight_norms.reshape(-1,1)

        return data


    def cost(self, x):
        """Returns 0 (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0


class tobi_prox_L2_and_ball(object):
    """This class defines the prox of the L2 norm on (all) alpha and indicator
    function on A lines so that they maintain the unit norm.
    Theorem 6.13 from: SIAM, Chapter 6, The Proximal Operator.
    First-Order Methods in Optimization, Amir Beck

    Parameters
    ----------
    proxVT: Matrix
        Corresponds to the VT matrix from RCA.
    beta_param: float number
        Corresponds to the beta (or lambda) parameter that goes with the fucn tion we will
        calculate the prox on. prox_{lambda f(.)}(y)
    iter: Integer
        Iteration number, just to follow track of the iterations. It could be part of the lambda
        update strategy for the prox calculation.
    """
    def __init__(self, proxVT):
        self.proxVT = proxVT
        self.beta_param = 0
        self.iter = 0

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        The extra factor is the beta_param!!
        Should be used on the proximal operator function
        """
        self.beta_param = extra_factor
        # print('Using L2 prox.. beta = %.5f it=%d\n'%(self.beta_param,self.iter))

        self.iter += 1 # not used in this prox
        # result = self.op_tobi_prox_l2_A(data)
        result = self.op_tobi_prox_L2_and_ball(data)
        # print('Prox results, it = %d'%(self.iter))
        # print(result)
        return result

    def op_tobi_prox_L2_and_ball(self, data):
        """ Apply the opterator.
        The operator can be used for the whole data matrix at once.
        """
        dividing_weigts =  np.ones(data.shape) + self.beta_param
        print('self.beta_param = %.5e'%(self.beta_param))
        print('RMS (myData) = %.5e'%(np.sqrt(np.mean( myData**2 ))))
        print('norm( myData) = %.5e\n'%(np.linalg.norm(myData)))
        myData = np.copy(data/dividing_weigts)
        weight_norms = np.sqrt(np.sum(myData.dot(self.proxVT)**2,axis=1))
        myData /= weight_norms.reshape(-1,1)

        return myData


    def cost(self, x):
        """Returns 0 (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0


class tobi_prox_elasticNet(object):
    """This class defines the elasticNet prox. The elastic net regularization considers the L2 + L1 norm
    Regularizations.
    « Parikh, N., Boyd, S. Proximal algorithms, Found. Trends Optim., V.1, N.3, 2014, pp.127-239. »
    ElasticNet reg: f(x) = |x|_1 + gamma/2 |x|_2^2
    Parameters
    ----------
    prox_weights: Matrix
        Corresponds to the weights of the weighted norm l_{w,2}. They are set by default to ones.
    beta_param: float number
        Corresponds to the beta (or lambda) parameter that goes with the fucn tion we will
        calculate the prox on. prox_{lambda f(.)}(y)
    gamma_param: float number
        Corresponds to the relative importance of the L2 norm compared to the L1 norm.
        A gamma of 2 corresponds to the typical importance distribution.
    iter: Integer
        Iteration number, just to follow track of the iterations. It could be part of the lambda
        update strategy for the prox calculation.
    """
    def __init__(self, prox_weights = None, gamma_param = 1.0, iter_max = 100):
        self.gamma_param = gamma_param
        self.prox_weights = prox_weights
        self.beta_param = 0
        self.iter = 0
        self.iter_max =iter_max

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        The extra factor is the beta_param!!
        Should be used on the proximal operator function
        """
        self.beta_param = extra_factor

        if self.prox_weights is None:
            self.prox_weights = np.ones(data.shape)

        self.iter += 1
        result = self.op_tobi_prox_elasticNet(data)
        print('Prox results, it = %d'%(self.iter))
        print(result)
        return result

    def op_tobi_prox_elasticNet(self, data):
        """ Apply the opterator on the whole data matrix
        for a vector: x = prox_{lambda || . ||^{2}_{w,2}}(y) =>
        x = prox_{lambda |.|_1}(x)* [ 1 / (1 + lambda w)]
        The operator can be used for the whole data matrix at once.
        """
        #self.beta_param = 0
        # print('Using elasticNet prox... it=%d\n'%(self.iter))
        dividing_weigts =  np.ones(data.shape) + self.beta_param * self.prox_weights * self.gamma_param
        # mat_out = self.op_tobi_prox_l1(mat=data,myThresh=self.beta_param)
        mat_out = self.op_tobi_prox_l1_elastic(mat=data,myThresh=0)

        return mat_out/dividing_weigts

    def op_tobi_prox_l1_elastic(self,mat=0,myThresh=0):
        """ Applies GMCA-soft-thresholding to each line of input matrix.

        Calls:

        * :func:`utils.gmca_thresh`

        """
        mat_out = np.copy(mat)
        shap = mat.shape
        for j in range(0,shap[0]):
            if myThresh == 0:
                # GMCA-like threshold calculation
                line = mat_out[j,:]
                idx = np.floor(len(line)*np.max([0.9-(self.iter/self.iter_max)*3,0.2])).astype(int)
                idx_thr = np.argsort(abs(line))[idx]
                thresh = abs(line[idx_thr])
            else:
                thresh = myThresh
            mat_out[j,:] = SoftThresholding(mat[j,:],thresh)
            # mat_out[j,:] = HardThresholding(mat_out[j,:],thresh)
        return mat_out

    def cost(self, x):
        """Returns 0 (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0


class tobi_prox_l21(object):
    """This class defines the joint l21 prox operator.
    « Mixed-norm estimates for the M/EEG inverse problem using accelerated gradient methods
Alexandre Gramfort, Matthieu Kowalski, Matti Hämäläinen »

    Parameters
    ----------
    prox_weights: Matrix
        Corresponds to the weights of the weighted norm l_{w,p}. They are set by default to ones.
    beta_param: float number
        Corresponds to the beta (or lambda) parameter that goes with the fucn tion we will
        calculate the prox on. prox_{lambda f(.)}(y)
    iter: Integer
        Iteration number, just to follow track of the iterations. It could be part of the lambda
        update strategy for the prox calculation.
    """
    def __init__(self, prox_weights = None, iter_max=50):
        self.prox_weights = prox_weights
        self.beta_param = 0
        self.iter = 0
        self.iter_max = iter_max

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        The extra factor is the beta_param!!
        Should be used on the proximal operator function
        """
        self.beta_param = extra_factor

        if self.prox_weights is None:
            self.prox_weights = np.ones(data.shape)

        self.iter += 1 # not used in this prox
        return self.op_tobi_prox_l21(data)

    def op_tobi_prox_l21(self, data):
        """ Apply the opterator on the whole data matrix
        for a vector: x = prox_{lambda || . ||^{2}_{w;2,1}}(y) =>
        x_{i,j}= y_{i,j}*(1 - [lambda*sqrt(w_i) / |y_i|_2 ])^{+}
        The operator can be used for the whole data matrix at once.
        """
        div_weigts =  np.sqrt(self.prox_weights)
        mat_out = np.copy(data)

        for it in range(data.shape[0]):
            mat_out[it,:] = self.line_op_prox_l21(mat_out[it,:],div_weigts[it,:])

        # print('\n prox(data):  \n')
        # print(mat_out)
        return mat_out

    def line_op_prox_l21(self, data_line, div_w_line):
        """ Apply the prox_l21 operator to one line of the data matrix.
        NEW_VERSION: Use the GMCA strategy to select the L1 (thresholding) parameters
        """
        line_l2norm = np.sqrt(np.sum(data_line**2))
        # print('\n line_l2norm = %f \n beta_param = %f \n norm/beta = %f'%(line_l2norm,self.beta_param,self.beta_param/line_l2norm))
        # print('\nData_line:\n')
        # print(data_line)

        # GMCA-like threshold calculation
        idx = np.floor(len(data_line)*np.max([0.8-(self.iter/self.iter_max)*3,0.3])).astype(int)
        idx_thr = np.argsort(abs(data_line))[idx]
        thresh = abs(data_line[idx_thr])

        # OLD VERSION
        # aux_vec = np.ones(data_line.shape) - div_w_line*(self.beta_param/line_l2norm)

        # NEW VERSION
        aux_vec = np.ones(data_line.shape) - div_w_line*(thresh/line_l2norm)

        # print('\nAux_vec:\n')
        # print(aux_vec)
        aux_vec[aux_vec<0] = 0
        # print('\n(Aux_vec)^+:\n')
        # print(aux_vec)
        # print('\Output line:\n')
        # print(data_line*aux_vec)

        return data_line*aux_vec

    def cost(self, x):
        """Returns 0 (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0

class StarletThreshold(object):
    """Apply soft thresholding in Starlet domain.

    Parameters
    ----------
    threshold: np.ndarray
        Threshold levels.
    thresh_type: str
        Whether soft- or hard-thresholding should be used. Default is ``'soft'``.
    """
    def __init__(self, threshold, thresh_type='soft'):
        self.threshold = threshold
        self._thresh_type = thresh_type

    def update_threshold(self, new_threshold, new_thresh_type=None):
        self.threshold = new_threshold
        if new_thresh_type in ['soft', 'hard']:
            self._thresh_type = new_thresh_type

    def op(self, transf_data, **kwargs):
        """Applies Starlet transform and perform thresholding.
        """
        # Threshold all scales but the coarse
        transf_data[:,:-1] = SoftThresholding(transf_data[:,:-1], self.threshold[:,:-1])
        return transf_data

    def cost(self, x, y):
        return 0 #TODO
