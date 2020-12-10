import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy.optimize as op
from scipy import special
import random
import emcee
import math
import ipdb
import pickle as pkl 
from pathlib import Path
import argparse


if __name__ == '__main__':
    filename = 'HI_JEWEL_etamax1_constmult_13invnbYJ'
    current_dir = Path.cwd()

    file = open( current_dir / (filename+'.pickle'), 'rb')
    datum = pkl.load(file)
    file.close()

    print(datum)
