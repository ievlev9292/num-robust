# Testing all the routines in this package
import numpy as np

# ## Testing fmin_robust_2d
from fmin_robust import fmin_robust_2d
# Simple quadratic with a large cutoff
x0_true = np.array([10, 0], dtype=float)
fun_test = lambda x: np.sum((x - x0_true)**2) if x[1] > -1 else np.nan
x0_test = np.array([100, 100])
x_min = fmin_robust_2d(fun_test, x0_test, rtol=1e-10, atol=1e-10, maxiter=np.inf, nodes_one_side=3, init_step=1e+4)
print(f'fmin_robust_2d: True min at {x0_true}, found min at {x_min}, the difference is {x0_true-x_min}\n')


### EXAMPLE OF USING narrow_to_domain and bisect_ignore_nan routines
from bisect_ignore_nan import narrow_to_domain, bisect_ignore_nan
def f(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        ans = np.log(x)
        # ans = np.log(1000 - x)
        # ans = np.log(x / 1e3)
        # ans = np.log(x) - np.log(1 - x)
        # ans = - np.log(x) - np.log(1 - x)
    return ans
max_iter = 10000
xtol = 1e-14
ignore_errors=False
full_output=True
x_left = -1000
x_right = 1000
print('Testing with a simple ln(x) function:')
## Testing narrow_to_domain
(x_l, x_r), iter_count = narrow_to_domain(x_left, x_right, f, xtol, max_iter, ignore_errors)
print(f'Expected domain [{max(0,x_left)}, {x_right}], found [{x_l}, {x_r}]')
## Testing bisect_ignore_nan
sol = bisect_ignore_nan(x_left, x_right, f, xtol, max_iter, full_output, ignore_errors)
print(f'Found root x={sol[0]} with uncertainty dx={sol[1]}\n')


# Testing utils
from utils import report_progress_time_simple, get_zero_crossing_ind, check_if_some_are_close

# Testing get_zero_crossing
a = [4,3,1,3,-3,-4]
true_zero_crossing = 3
found_zero_crossing = get_zero_crossing_ind(a)
print(f'Testing get_zero_crossing: true_zero_crossing={true_zero_crossing}, found_zero_crossing={found_zero_crossing}\n')

# Testing check_if_some_are_close
a = [0, 1, -2, 0.1]
tol = 0.02
print(f'Testing check_if_some_are_close: expected None, got {check_if_some_are_close(a, tol)}')
tol = 0.2
print(f'Testing check_if_some_are_close: expected (0, 3), got {check_if_some_are_close(a, tol)}')
print('\n')


