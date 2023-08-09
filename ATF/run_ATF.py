import pickle
import sys

import ATFPython

# # get the pickled input arguments (if any)
# inputs = pickle.loads(sys.stdin.buffer.read())

# run ATF
result = ATFPython.tune()
result['T'] = result['T'] * 1e-9 # nanoseconds to seconds

# write the result to stdout
sys.stdout.buffer.write(pickle.dumps(result))
