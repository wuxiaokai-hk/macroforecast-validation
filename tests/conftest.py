import os
import sys

# the developer's repo (for the transform cross-check) sits next to this one
DEV = os.path.join(os.path.dirname(__file__), "..", "..", "Forecasting-test")
if os.path.isdir(DEV):
    sys.path.insert(0, os.path.abspath(DEV))
