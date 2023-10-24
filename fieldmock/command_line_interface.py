import lpt
import fieldmock
from fieldmock.util import FieldmockDefaults

def main():

    # for full res websky 1LPT at nside=1024:
    #   export LPT_DISPLACEMENTS_PATH=/pscratch/sd/m/malvarez/websky-displacements/
    #   srun -n $nproc --gpus-per-task=1 xgfieldmock --N 6144 --Nside 1024 --ityp lptfiles

    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Commandline interface to fieldmock')

    parser.add_argument('modelID', type=str)
    parser.add_argument('--N',     type=int, help=f'grid dimension [default = {FieldmockDefaults["N"]}]', 
                        default=FieldmockDefaults["N"])
    parser.add_argument('--Lbox',  type=int, help=f'box size in Mpc [default = {FieldmockDefaults["Lbox"]}]', 
                        default=FieldmockDefaults["Lbox"])
    parser.add_argument('--Nside', type=int, help=   f'output map healpix Nside [default = {FieldmockDefaults["Nside"]}]', 
                        default=FieldmockDefaults['Nside'])
    parser.add_argument('--ityp',    type=str, help=f'field input type [default = {FieldmockDefaults["ityp"]}]',
                        default=FieldmockDefaults['ityp'])
    args = parser.parse_args()

    ID    = args.modelID
    N     = args.N
    Lbox  = args.Lbox
    nside = args.Nside
    ityp  = args.ityp

    fieldsky = fieldmock.FieldSky(ID=ID,N=N,Lbox=Lbox,Nside=Nside,ityp=ityp)

    return fieldsky.generate()

if __name__ == "__main__":
    import sys
    sys.exit(main())