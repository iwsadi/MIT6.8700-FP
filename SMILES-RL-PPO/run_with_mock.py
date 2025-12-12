#!/usr/bin/env python
"""
Wrapper script that mocks OpenEye modules before running training.
OpenEye is a commercial package not needed for basic scoring functions.
"""
import sys
import os
import warnings
from unittest.mock import MagicMock

# Suppress joblib parallel output (from sklearn RandomForest)
# This must be done before importing sklearn
os.environ['JOBLIB_VERBOSITY'] = '0'
os.environ['LOKY_PICKLER'] = 'pickle'

# Monkey-patch joblib to suppress all parallel output
import joblib
_original_parallel_init = joblib.Parallel.__init__
def _quiet_parallel_init(self, *args, **kwargs):
    kwargs['verbose'] = 0
    return _original_parallel_init(self, *args, **kwargs)
joblib.Parallel.__init__ = _quiet_parallel_init

# Suppress RDKit deprecation warnings (MorganGenerator)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Suppress all RDKit warnings

# Also suppress Python deprecation warnings from RDKit
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*MorganGenerator.*')

# Mock the openeye module before any imports
sys.modules['openeye'] = MagicMock()
sys.modules['openeye.oechem'] = MagicMock()
sys.modules['openeye.oeomega'] = MagicMock()
sys.modules['openeye.oeshape'] = MagicMock()

# Now import and run the main function
from run import main

if __name__ == "__main__":
    main()
