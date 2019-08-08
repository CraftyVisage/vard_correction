'''Calculates correction to the upper bound using blocked trajectory data. Uses
the scaled cumulant generating function representation to get this correction
as described in:

Jacobson, D. & Whitelam, S. Direct evaluation of dynamical large-deviation 
rate functions using a variational ansatz. arXiv:1903.06098 [cond-mat] (2019).

See also:

Rohwer, C. M., Angeletti, F. & Touchette, H. Convergence of large-deviation 
estimators. Phys. Rev. E 92, 052104 (2015).

For data formatting info see command line args and the read_data method.
'''

import argparse
import numpy as np
import scipy.special
import IPython as ipy
import matplotlib.pyplot as plt

def calculate_scgf_slice(k_array, observable_array, block_size, k_s = 0.0,
                         s_array = None, chunk_size = 10000 * 6000):
    '''Calculates slice of 2d scgf where one coordinate is fixed.

    Takes:
        k_array: array of ks to calculate scgf at
        observable_array: array of blocked intensive observable values
        block_size: size of block (in time or events) used to calculate each
            observable
        k_s: value of second coordinate of slice
        s_array: observable associated with second coordinate
        chunk_size: split k_array into chunks of this size to avoid running out
            of memory
    Returns:
        scgf_slice: slice of 2d scgf along k_s
    '''
    if type(k_array) == np.float64:
        # make work with single numbers
        k_array = np.array([k_array])
    scgf_slice = []
    N_observables = observable_array.size
    # prevent memory overflow by splitting into chunks
    chunks = np.max((1, int(np.ceil(N_observables * k_array.size /
                                    (chunk_size)))))
    k_chunks = np.array_split(k_array, chunks)
    for i, k_chunk in enumerate(k_chunks):
        # row i is a_array * k_array[i]
        tmp = np.outer(k_chunk, observable_array)
        if s_array is not None:
             tmp += k_s * s_array
        tmp = tmp * block_size
        # need to use special function for improved numerical stability
        sub_scgf_slice = scipy.special.logsumexp(tmp, axis = 1,
                                                 b = 1.0 / N_observables)
        sub_scgf_slice = (1.0 / block_size) * sub_scgf_slice
        scgf_slice += list(sub_scgf_slice)
    scgf_slice = np.array(scgf_slice)
    return scgf_slice

def read_data(args):
    '''Reads data for correction.

    Data format is two columns, with one row for each block. The # symbol is
    treated as a comment and is ignored. The first column is

    \delta_a = a - a_0, 

    the time intensive amount of the observable accumulated within that block 
    minus the mean. The second column is 

    \delta q = q - q_0

    the time intensive value of the likelihood factor minus the mean.

    Takes:
        args: command line args
    Returns:
        blocked_delta_a_array: blocked values of intensive observable deltas
        blocked_delta_q_array: blocked values of intensive likelihood factor
            deltas
    '''
    # 0:delta_a 1:delta_q
    data = np.loadtxt(args.input, skiprows = 1)
    blocked_delta_a_array = data[:, 0]
    blocked_delta_q_array = data[:, 1]
    return blocked_delta_a_array, blocked_delta_q_array

def get_correction(args, delta_a_sets, delta_q_sets, k_delta_a_array):
    '''Calculates correction to bound using scgf data.

    Takes:
        args: command line parameters
        delta_a_sets: list of sets of intensive blocked delta a values
        delta_q_sets: list of sets of intensive blocked delta q values
        k_delta_a_array: k_delta_a values to scan to find correction point 
            k_delta_q = 1.0 and delta a = 0
    Returns:
        average_correction: estimated value of correction from data sets
        correction_error: estimated statistical error from variance of 
            data sets
        average_rate_function: value of rate function at correction
        rate_function_error: estimated statistical error from variance of 
            data sets
        max_convergence_rate_function: estimated max value of rate function 
            for which correction estimator will converge
    '''
    # the values of these quantities at the point needed to get the correction
    # for each data set
    correction_k_delta_a_array = []
    # this should be all zeros up to the accuracy of the k_delta_a_array
    correction_delta_a_array = []
    correction_delta_q_array = []
    correction_scgf_array = []
    correction_array = []
    correction_rate_function_array = []
    
    # for each data set calculate value of correction
    # get error by looking at distribution of correction values over sets
    for s in range(args.N_data_sets):
        print("Processing set " + str(s + 1))
        blocked_delta_a_array = delta_a_sets[s]
        blocked_delta_q_array = delta_q_sets[s]
        
        # calculate a slice of the 2d scgf
        # slice has k_delta_q = 1.0 while k_delta_a varies
        # want to find point that corresponds to delta_a = 0
        k_delta_q = 1.0
        scgf_slice = calculate_scgf_slice(k_delta_a_array,
                                          blocked_delta_a_array,
                                          args.block_time,
                                          k_delta_q,
                                          blocked_delta_q_array,
                                          args.chunk_size)
        delta_a_array = np.gradient(scgf_slice, k_delta_a_array)
        # find a = a_mean
        delta_a_zero_index = np.argmin(np.abs(delta_a_array))
        correction_delta_a_array.append(delta_a_array[delta_a_zero_index])
        correction_k_delta_a = k_delta_a_array[delta_a_zero_index]
        correction_k_delta_a_array.append(correction_k_delta_a)
        scgf_value = scgf_slice[delta_a_zero_index]
        correction_scgf_array.append(scgf_value)
    
        # now need to find derivative with respect to k_delta_q at
        # the point k_delta_a = correction_k_delta_a, k_delta_q = 1.0
        # this gives value of delta q associated with that point
        perturbed_k_delta_q = k_delta_q + args.epsilon
        perturbed_scgf_value = calculate_scgf_slice(correction_k_delta_a,
                                                    blocked_delta_a_array,
                                                    args.block_time,
                                                    perturbed_k_delta_q,
                                                    blocked_delta_q_array,
                                                    args.chunk_size)
        perturbed_scgf_value = perturbed_scgf_value[0]
        correction_delta_q = ((perturbed_scgf_value - scgf_value) /
                                (perturbed_k_delta_q - k_delta_q))
        correction_delta_q_array.append(correction_delta_q)
    
        # now use double legendre transform to recover rate function value
        # first term is zero, including for clarity, note the value of
        # correction_k_delta_a still needs to be calculated in order to get
        # correction_delta_q
        correction_delta_a = 0
        rate_function = (correction_k_delta_a * correction_delta_a +
                         k_delta_q * correction_delta_q -
                         scgf_value)
        correction_rate_function_array.append(rate_function)
        correction = correction_delta_q - rate_function
        correction_array.append(correction)
    correction_k_delta_a_array = np.array(correction_k_delta_a_array)
    correction_delta_a_array = np.array(correction_delta_a_array)
    correction_delta_q_array = np.array(correction_delta_q_array)
    correction_scgf_array = np.array(correction_scgf_array)
    correction_rate_function_array = np.array(correction_rate_function_array)
    correction_array = np.array(correction_array)

    # throw error if k_a_start and k_a_end were not wide enough
    k_delta_a_buffer = np.diff(k_delta_a_array)[0] * 100
    min_correction_k_delta_a = np.min(correction_k_delta_a_array)
    if min_correction_k_delta_a - args.k_delta_a_start < k_delta_a_buffer:
        print("ERROR: decrease k_delta_a_start!")
        print("k_delta_a_start = " + str(args.k_delta_a_start))
        print("min_correction_k_delta_a = " +
              str(min_correction_k_delta_a))
        exit()
    max_correction_k_delta_a = np.max(correction_k_delta_a_array)
    if args.k_delta_a_end - max_correction_k_delta_a < k_delta_a_buffer:
        print("ERROR: increase k_delta_a_end!")
        print("k_delta_a_end = " + str(args.k_delta_a_end))
        print("max_correction_k_delta_a = " +
              str(max_correction_k_delta_a))
        exit()
    average_correction = np.mean(correction_array)
    correction_error = np.std(correction_array) / np.sqrt(args.N_data_sets)
    average_rate_function = np.mean(correction_rate_function_array)
    rate_function_error = (np.std(correction_rate_function_array) /
                           np.sqrt(args.N_data_sets))

    # estimate convergence
    k_delta_a = np.mean(correction_k_delta_a_array)
    k_delta_q = 1.0
    max_convergence_rate_function = check_convergence(k_delta_a, k_delta_q,
                                                      delta_a_sets, 
                                                      delta_q_sets,
                                                      args)
    return (average_correction, correction_error,
            average_rate_function, rate_function_error,
            max_convergence_rate_function)

def check_convergence(k_delta_a, k_delta_q, delta_a_sets, delta_q_sets, args):
    '''Checks to see if empirical scgf estimator converges at specified point.

    Takes:
        k_delta_a: k_delta_a value of correction point
        k_delta_q: k_delta_q_value of correction point
        delta_a_sets: list of sets of intensive blocked delta a values
        delta_q_sets: list of sets of intensive blocked delta q values
        args: command line parameters
    Returns:
        max_rate_function: estimate of max rate function value for which 
            estimator will converge
    '''
    scale_factor = args.linearization_scale_factor
    points = args.linearization_points
    # used for calculating derivatives with finite difference, make sure
    # answer does not change when this gets smaller
    epsilon = args.epsilon

    # first get value of rate function where estimator stops converging along
    # k_delta_a and k_delta_q axes
    # k_delta_a axis first
    endpoint = max(np.sign(k_delta_a), scale_factor * k_delta_a, key = abs)
    k_delta_a_array = np.linspace(0.0, endpoint, points)
    k_delta_a_array = np.sort(k_delta_a_array)
    # return is 0:observable_array 1:observable_errors 2:scgf_array
    # 3:scgf_errors
    tmp = calculate_convergence_slice(k_delta_a_array, delta_a_sets,
                                      args.block_time, args.epsilon,
                                      args.chunk_size)
    delta_a_array = tmp[0]
    delta_a_errors = tmp[1]
    convergence_index = np.argmax(delta_a_errors)
    scgf_array = tmp[2]
    rate_function = (k_delta_a_array[convergence_index] *
                     delta_a_array[convergence_index])
    rate_function -= scgf_array[convergence_index]
    max_rate_function = rate_function

    # now k_delta_q axis
    endpoint = max(np.sign(k_delta_q), scale_factor * k_delta_q, key = abs)
    k_delta_q_array = np.linspace(0.0, endpoint, points)
    k_delta_q_array = np.sort(k_delta_q_array)
    # return is 0:observable_array 1:observable_errors 2:scgf_array
    # 3:scgf_errors
    tmp = calculate_convergence_slice(k_delta_q_array, delta_q_sets,
                                      args.block_time, args.epsilon,
                                      args.chunk_size)    
    delta_q_array = tmp[0]
    delta_q_errors = tmp[1]
    convergence_index = np.argmax(delta_q_errors)
    scgf_array = tmp[2]
    rate_function = (k_delta_q_array[convergence_index] *
                     delta_q_array[convergence_index])
    rate_function -= scgf_array[convergence_index]
    max_rate_function = np.min((max_rate_function, rate_function))
    return max_rate_function

def calculate_convergence_slice(k_array, observable_sets, block_time, epsilon,
                                chunk_size):
    '''Calculates convergence slice of scgf along axis where second k 
    coordinate is 0.0.

    Errors calculated from variance of each repetition.
    
    Takes:
        k_array: array of k value to calculate over
        observable_sets: list of sets of intensive blocked observable values
        block_time: time length of each block (block_size * scgf_interval)
        epsilon: small value used for calculating finite difference derivatives
        chunk_size: split k_array into chunks of this size to avoid running out
            of memory
    Returns:
        observable_array: observable value at each k point
        observable_errors: standard error 
        scgf_array: scgf at each k point
        scgf_errors: standard error
    '''
    N_sets = len(observable_sets)
    points = k_array.size

    # for each set, calculate observable at each k point
    observable_arrays = np.zeros((N_sets, points))
    scgf_arrays = np.zeros((N_sets, points))
    for s in range(N_sets):
        blocked_observable_array = observable_sets[s]
        scgf = calculate_scgf_slice(k_array, blocked_observable_array,
                                    block_time, chunk_size = chunk_size)
        scgf_arrays[s, :] = scgf
        # perturb k_array by small amount
        perturbed_scgf = calculate_scgf_slice(k_array + epsilon,
                                              blocked_observable_array,
                                              block_time,
                                              chunk_size = chunk_size)
        observable_arrays[s, :] = ((perturbed_scgf - scgf_arrays[s, :]) /
                                   epsilon)
    observable_array = np.mean(observable_arrays, axis = 0)
    observable_errors = np.std(observable_arrays, axis = 0) / np.sqrt(N_sets)
    scgf_array = np.mean(scgf_arrays, axis = 0)
    scgf_errors = np.std(scgf_arrays, axis = 0) / np.sqrt(N_sets)
    return (observable_array, observable_errors, scgf_array, scgf_errors)

def get_args():
    '''Gets command line args.

    Returns:
        args: command line arguments
    '''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = formatter)
    parser.add_argument("block_time", type =float, 
                        help = "the length of the blocks, increase this to "
                        "obtain convergence")
    parser.add_argument("--input", default = "correction_data.txt",
                        help = "name of data file")
    parser.add_argument("--N_data_sets", type = int, default = 5,
                        help = "use this to split data into multiple sets to "
                        "calculate statistical errors. See equations 30 and "
                        "31.")
    parser.add_argument("--k_delta_a_start", type = float, default = -3.0,
                        help = "start value for k_delta_a scan used to find "
                        "k_\tilde{a}_0. See equation 28 in paper.")
    parser.add_argument("--k_delta_a_end", type = float, default = 3.0,
                        help = "end value for k_delta_a scan used to find "
                        "k_\tilde{a}_0. See equation 28 in paper.")
    parser.add_argument("--k_delta_a_points", type = int, default = 6001,
                        help = "points to use between k_a_start and k_a_end. "
                        "Convergence parameter, increase until does not "
                        "affect results.")
    parser.add_argument("--linearization_scale_factor", type = float,
                        default = 4.0,
                        help = "controls the region over which k is scanned "
                        "for the linearization check. If too small the check "
                        "may report an artifically low max convergence value.")
    parser.add_argument("--linearization_points", type = int, default = 50,
                        help = "controls the resolution of linearization " +
                        "check calculation. If too small the check may " +
                        "report an artifically low max convergence value.")
    parser.add_argument("--alpha", type = float, default = 0.8,
                        help = "empirical constant used in convergence check."
                        " See equation 34 in paper.")
    parser.add_argument("--epsilon", type = float, default = 1e-6,
                        help = "used for calculating derivatives with finite "
                        "difference. Convergence parameter.")
    parser.add_argument("--chunk_size", type = int, default = 10000 * 6000,
                        help = "decrease this if memory overflows.")

    args = parser.parse_args()
    return args

def main():
    '''Coordinates calculation.'''
    args = get_args()

    blocked_delta_a_array, blocked_delta_q_array = read_data(args)
    print("Found " + str(blocked_delta_a_array.size) + " blocks")
    # split total blocks into this many separate data sets to calculate
    # statistical error
    print("Splitting into " + str(args.N_data_sets) + " data sets of " +
          str(blocked_delta_a_array.size // args.N_data_sets) + " blocks")
    delta_a_sets = np.split(blocked_delta_a_array, args.N_data_sets)
    delta_q_sets = np.split(blocked_delta_q_array, args.N_data_sets)
    
    k_delta_a_array = np.linspace(args.k_delta_a_start, args.k_delta_a_end,
                                  args.k_delta_a_points)
    
    tmp = get_correction(args, delta_a_sets, delta_q_sets, k_delta_a_array)
    # correction: correction to the bound 
    correction = tmp[0]
    # correction_error: standard error computed from sqrt(variance / N_trials)
    correction_error = tmp[1]
    # rate_function: value of rate function at correction    
    rate_function = tmp[2]
    # rate_function_error: standard error computed from
    # sqrt(variance / N_trials)
    rate_function_error = tmp[3]
    # max_rate_function: estimated max value of rate function for which
    # estimator will converge, if this is larger than rate_function estimate
    # cannot be trusted
    max_rate_function = tmp[4]

    print("\ncorrection = {} +- {}".format(correction, correction_error))
    print("rate function = {} +- {}".format(rate_function,
                                            rate_function_error))
    print("max rate function = {}\n".format(max_rate_function))
    if rate_function > args.alpha * max_rate_function:
        print("WARNING: rate_function > alpha * max_rate_function, cannot " +
              "trust estimate")
    
if __name__ == "__main__":
    main()
