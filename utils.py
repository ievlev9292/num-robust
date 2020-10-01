####################################################################################
# by Free Fall  http://inspirehep.net/author/profile/E.Ievlev.1
# Started developing 2018 November
####################################################################################


import time
import numpy as np

'''
    Progress-related
'''

def report_progress_time_simple(current_iter, tot_iter, start_time):
    """
    A simple function
    Report how much work was done and estimate remaining time
    Example of usage: see below.
    """
    if current_iter != 0:
        done_ratio = current_iter / tot_iter
        elapsed_time = (time.time() - start_time) / 60
        estimated_time_left = elapsed_time * (1 / done_ratio - 1)
        print("Done {:.2f} percent ({:d} out of {:d}), elapsed time {:.2f} minutes, estimated time left {:.2f} minutes ...".format(100 * done_ratio, current_iter, tot_iter, elapsed_time, estimated_time_left))
    return None

# # Example of using report_progress_time_simple
# ti = 7
# st = time.time()
# for i in range(ti):
#     time.sleep(1.3)
#     report_progress_time_simple(i, ti, st)

'''
    Numerical utils
'''

def get_zero_crossing_ind(a):
    """
    Find the first zero crossing in the array a
    :param a: array (without NaN)
    :return: index i such that a[i] and a[i+1] have different signs
    """
    zero_crossings_ind = np.where(np.diff(np.sign(a)))[0]
    # get the first one
    try:
        the_ind = zero_crossings_ind[0]
    except IndexError:
        the_ind = np.nan
    return the_ind


def check_if_some_are_close(a, tol):
    """
    Check if there is a pair of close elements in the iterable a
    :param a: 1d iterable
    :param tol: absolute and relative tolerance
    :return:
        if there is a pair of elements that are close with tolerance tol,
        then return a tuple (i1, i2) if indices of such elements
        (only the first leftmost pair is returned)
        else return None
    """
    are_close = False
    n = len(a)
    for i in range(n-1):
        for j in range(i+1, n):
            if np.isclose(a[i], a[j], rtol=tol, atol=tol):
                are_close = True
                return i, j
    return None



