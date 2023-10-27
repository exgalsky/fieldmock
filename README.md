# xgfield

Generation of mocks from a field representation of LSS on the observer's past light cone.

## Installation
1. git clone https://github.com/exgalsky/xgfield.git
2. cd xgfield
3. pip install .

## Running
Currently runs on perlmutter in the [xgsmenv](https://github.com/exgalsky/xgsmenv) enviroment.

Mocks can be generated through the command line interface on a login node in serial with a single CPU (Perlmutter non-login nodes were down at the time of writing):
```
% module use /global/cfs/cdirs/mp107/exgal/env/xgsmenv/20231013-0.0.0/modulefiles/
% module load xgsmenv
% export LPT_DISPLACEMENTS_PATH=/pscratch/sd/m/malvarez/websky-displacements/
% mkdir -p output 
% xgfield fieldsky-test-768-lptfiles --N 768 --Nside 1024 --no-gpu --no-mpi # reading from external LPT field
% python xgfield/scripts/example.py fieldsky-test-768-cube                  # passes in same field via cube object
% stat -c "%n,%s" output/kappa_fieldsky-test-768-* | column -t -s,
output/kappa_fieldsky-test-768-cube-768_nside-1024.fits      100670400
output/kappa_fieldsky-test-768-lptfiles-768_nside-1024.fits  100670400
```
