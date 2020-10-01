####################################################################################
# by Free Fall  http://inspirehep.net/author/profile/E.Ievlev.1
# Started developing 2018 November
####################################################################################

import numpy as np
import inspect


def narrow_to_domain(x_left, x_right, f, xtol, max_iter, ignore_errors):
    """
    Narrows the domain of f(..) on [x_left, x_right]
    Assumes that the narrowing of the domain of f(..) on [x_left, x_right] is a connected set
    :param x_left:
    :param x_right:
    :param f: function with signature f(scalar) -> real scalar
    :param xtol: absolute tolerance
    :param max_iter: (roughly) maximum number of the f(..) function calls
    :param ignore_errors: if True, do not print warnings and error messages.
            Note that in case of an error the function returns np.nan
    :return: ((x_left, x_right), iter_count) where
            (x_left, x_right): the required interval
            iter_count: (roughly) the number of the f(..) function calls
    """
    f_left = f(x_left)
    f_right = f(x_right)
    iter_count = 0
    # this function is used later to determine the boundaries of the f domain:
    f_domain = lambda x: -1 if np.isnan(f(x)) else 1
    # if the domain is completely unknown, let's try to determine it
    if (np.isnan(f_left) and np.isnan(f_right)):
        reached_xtol = x_right - x_left
        found_anchor = False
        anchor_point = np.nan
        while not (found_anchor or (reached_xtol < xtol) or (iter_count > max_iter)):
            reached_xtol /= 5
            iter_count += int((x_right - x_left) / reached_xtol)
            for x in np.arange(x_left, x_right, reached_xtol):
                if not np.isnan(f(x)):
                    found_anchor = True
                    anchor_point = x
                    break
        if not found_anchor:
            if not ignore_errors:
                print('Error in {}: domain of the function was not determined'.format(inspect.stack()[0][3]))
            return (np.nan, np.nan), iter_count
        # if the anchor point was determined, find right boundary more precisely
        reached_xtol *= 5
        anchor_point, reached_xtol, did_converge, new_iter_count = bisect_ignore_nan(anchor_point, anchor_point + reached_xtol, f_domain, xtol, max_iter - iter_count, full_output=True, ignore_errors=ignore_errors)
        iter_count += new_iter_count
        x_right = anchor_point - reached_xtol / 1.9
        f_right = f(x_right)
        if not did_converge:
            if not ignore_errors:
                print('Error in {}: reached max_iter'.format(inspect.stack()[0][3]))
            return (np.nan, np.nan), iter_count
    # If f=np.nan on only one boundary, determine the other one
    if np.isnan(f_left):
        anchor_point, reached_xtol, did_converge, new_iter_count = bisect_ignore_nan(x_left, x_right, f_domain, xtol=xtol, max_iter=(max_iter - iter_count), full_output=True, ignore_errors=ignore_errors)
        iter_count += new_iter_count
        x_left = anchor_point + reached_xtol / 1.9
        if not did_converge:
            if not ignore_errors:
                print('Error in {}: reached max_iter'.format(inspect.stack()[0][3]))
            return (np.nan, np.nan), iter_count
    if np.isnan(f_right):
        anchor_point, reached_xtol, did_converge, new_iter_count = bisect_ignore_nan(x_left, x_right, f_domain, xtol, max_iter - iter_count, full_output=True, ignore_errors=ignore_errors)
        iter_count += new_iter_count
        x_right = anchor_point - reached_xtol / 1.9
        if not did_converge:
            if not ignore_errors:
                print('Error in {}: reached max_iter'.format(inspect.stack()[0][3]))
            return (np.nan, np.nan), iter_count
    return (x_left, x_right), iter_count



def bisect_ignore_nan(x_left, x_right, f, xtol=1e-14, max_iter=1000, full_output=False, ignore_errors=False):
    """
    Bisect routine for solving the equation f(x)=0
    Allows for f(x) to return NaN
    Assumes whatever the function narrow_to_domain(..) assumes.
    :param x_left:
    :param x_right: start search in the interval [x_left, x_right]
    :param f: the function
    :param xtol: tolerance of x, absolute and relative.
            The convergence is considered reached if [reached_xtol < max(np.abs(x_mid) * xtol, xtol)]
    :param max_iter: (roughly) maximum number of the f(..) function calls
    :param full_output: if True, provide full output
    :param ignore_errors: if True, do not print warnings and error messages.
            Note that in case of an error the function returns np.nan
    :return:
        if full_output: returns tuple (x_mid, reached_xtol, did_converge, iter_count) where
            x_mid: found solution
            reached_xtol: tolerance of the solution (might be > xtol if e.g. max_iter was reached)
            did_converge: True if xtol was reached
            iter_count: (roughly) number of the used f(..) function calls
        else:
            x_mid: found solution
    """
    # First, calculate the initial function values
    f_left = f(x_left)
    f_right = f(x_right)
    iter_count = 2
    # Check for trivial cases
    if f_left == 0:
        return x_left
    if f_right == 0:
        return x_right
    # If some boundary values are NaN, then pass to narrow_to_domain(..)
    if np.isnan(f_left) or np.isnan(f_right):
        (x_left, x_right), iter_count_narrowing = narrow_to_domain(x_left, x_right, f, xtol, max_iter, ignore_errors)
        iter_count += iter_count_narrowing
        if np.isnan(x_left) or np.isnan(x_right):
            return np.nan
        f_left = f(x_left)
        f_right = f(x_right)
    # Check for consistency
    if f_left * f_right > 0:
        if not ignore_errors:
            print('Error in {}: values of the function on two boundaries are both of the same sign'
                  .format(inspect.stack()[0][3]))
        return np.nan
    # Start the main loop
    reached_xtol = x_right - x_left
    x_mid = (x_right + x_left) / 2
    while not ((reached_xtol < max(np.abs(x_mid) * xtol, xtol)) or (iter_count > max_iter)):
        x_mid = (x_right + x_left) / 2
        f_mid = f(x_mid)
        if f_mid * f_right > 0:
            x_right, f_right = x_mid, f_mid
        else:
            x_left, f_left = x_mid, f_mid
        reached_xtol = x_right - x_left
        iter_count += 1
    # Check convergence
    did_converge = reached_xtol < max(np.abs(x_mid) * xtol, xtol)
    if not did_converge and not ignore_errors:
        print('Warning in {}: convergence was not reached at max_iter={}, reached_xtol={}'
              .format(inspect.stack()[0][3], max_iter, reached_xtol))
    # Get the found solution and return
    x_mid = (x_right + x_left) / 2
    if full_output:
        return x_mid, reached_xtol, did_converge, iter_count
    else:
        return x_mid

