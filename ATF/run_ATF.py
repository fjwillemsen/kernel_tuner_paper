import pickle
import sys

import ATFPython

# # get the pickled input arguments (if any)
# inputs = pickle.loads(sys.stdin.buffer.read())

# run ATF
result = ATFPython.add(1, 2)

# write the result to stdout
to_return = {'a':1, 'b': 3, 'res': result}
sys.stdout.buffer.write(pickle.dumps(to_return))
