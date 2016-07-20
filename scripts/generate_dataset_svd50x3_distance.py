from utilities import SimpleTransform, identity
from generate_dataset import generate_dataset
import numpy as np
__author__ = 'mudit'

# t_fn = np.sqrt
t_fn = lambda x: np.log(1+x)
# t_fn = identity


