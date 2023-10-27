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

% script=xgfield/scripts/cube_example.py
% cubecoms="python $script"
% filecoms="xgfield"
% $filecomp="srun -n 4 --gpus-per-task=1 $filecoms"
% $cubecomp="srun -n 4 --gpus-per-task=1 $cubecoms"
%
% $filecoms fieldsky-test-files --no-mpi # displacements from external file processed in serial
% $cubecoms fieldsky-test-cubes   serial # displacements from external cube processed in serial
% $filecomp fieldsky-test-filep          # displacements from external cube processed in parallel
% $cubecomp fieldsky-test-cubep parallel # displacements from external cube processed in parallel

% stat -c "%n,%s" output/kappa_fieldsky-test-* | column -t -s,
output/kappa_fieldsky-test-cubep-768_nside-1024.fits  100670400
output/kappa_fieldsky-test-cubes-768_nside-1024.fits  100670400
output/kappa_fieldsky-test-filep-768_nside-1024.fits  100670400
output/kappa_fieldsky-test-files-768_nside-1024.fits  100670400

% fitsdiff -q -a 1e-5 output/kappa_fieldsky-test-files-768_nside-1024.fits output/kappa_fieldsky-test-cubep-768_nside-1024.fits ; echo $?
0
```
