import lpt
from xgfield import fieldsky
import sys
import numpy as np
import os

N      = 768
Lbox   = 7700.
Nside  = 1024
input  = 'cube'
usegpu = False
usempi = False

try:
    path2disp = os.environ['LPT_DISPLACEMENTS_PATH']
except:
    path2disp = '/Users/shamik/Documents/Work/websky_datacube/'

cube = lpt.Cube(N=N,partype=None)

cube.s1x = np.fromfile(path2disp+'sx1_7700Mpc_n6144_nb30_nt16_no768',dtype=np.float32)
cube.s1y = np.fromfile(path2disp+'sy1_7700Mpc_n6144_nb30_nt16_no768',dtype=np.float32)
cube.s1z = np.fromfile(path2disp+'sz1_7700Mpc_n6144_nb30_nt16_no768',dtype=np.float32)
cube.s2x = cube.s1x * 0.
cube.s2y = cube.s1x * 0.
cube.s2z = cube.s1x * 0.

def _run_example(usegpu,usempi):
    sky = fieldsky.FieldSky(ID    = sys.argv[1],
                            N     = N,
                            Lbox  = Lbox,
                            Nside = Nside,
                            input = input,
                            gpu   = usegpu,
                            mpi   = usempi,
                            cube  = cube)

    sky.generate()

_run_example(False,False)
