####################################################################################
# by Free Fall  http://inspirehep.net/author/profile/E.Ievlev.1
# Started developing 2018 November
####################################################################################

import numpy as np


def fmin_robust_2d(fun, x0, rtol=1e-8, atol=1e-8, maxiter=np.inf, nodes_one_side=3, init_step=1e-1):
    """
    Find local minimum of a function of two variables.
    Similar to scipy.optimize.fmin but should be more robust.
    At each step it looks around in the vicinity of current guess
    by computing  f(x) at x placed on sides of a square
    centered at current guess.
    Then it goes to the point where f(x) is minimal.
    Note: it is advisory to take the initial guess x0 far from the region in which fun returns nans,
    but still close to the true solution.
    :param fun: callable f(x)
    :param x0: initial guess
    :param rtol: relative x-tolerance
    :param atol: absolute x-tolerance
    :param maxiter: maximum number of f(..) function calls
    :param nodes_one_side: number of nodes at each side of the vicinity square to be checked
    :param init_step: initial size of the vicinity
    :return: the minimum, if found; otherwise, an array of np.nans.
    """
    # preliminary checks
    fun_center = fun(x0)
    if np.isnan(fun_center):
        return x0 * np.nan
    # Prepare the mesh for the exploration of the vicinity
    possible_dx = np.linspace(- (nodes_one_side//2), nodes_one_side//2, nodes_one_side)
    unit_square_zero_centered = [[d1, d2] for d1 in possible_dx for d2 in possible_dx]
    try:
        unit_square_zero_centered.remove([0, 0])
    except ValueError:
        pass
    # unit_square_zero_centered = [np.array(pair, dtype=float) for pair in unit_square_zero_centered]
    unit_square_zero_centered = np.array(unit_square_zero_centered)
    # MAIN LOOP
    cur_step = init_step
    STEP_REDUCTION_COEFFICIENT = 2
    success_flag = False
    iter_counter = 0
    while cur_step > max(atol, rtol*np.linalg.norm(x0)) and iter_counter < maxiter:
        vicinity_points = x0 + unit_square_zero_centered * cur_step
        # go over the points in the vicinity
        fun_center = fun(x0)
        fun_vicinity_vals = [fun(x) for x in vicinity_points]
        iter_counter += len(fun_vicinity_vals) + 1
        if np.any(np.isnan(fun_vicinity_vals)):
            # probably we are near the boundary of the domain
            # reduce step, proceed to next iter
            cur_step /= STEP_REDUCTION_COEFFICIENT
            success_flag = False
            continue
        # if there are no np.nans in the vicinity, search for the minimum
        fun_vicinity_argmin = np.argmin(fun_vicinity_vals)
        if fun_center > fun_vicinity_vals[fun_vicinity_argmin]:
            # great, then go to that point
            x0 = vicinity_points[fun_vicinity_argmin]
            # just in case, to avoid infinite looping slightly change the step
            cur_step *= 0.999
        else:
            # looks like we are in the minimum
            # nothing interesting in the vicinity, reduce step
            cur_step /= STEP_REDUCTION_COEFFICIENT
            success_flag = True
    # when the loop is over, see if some minimum was found
    if success_flag:
        return x0
    else:
        # otherwise return nan
        return x0 * np.nan
