def main():

    # for full res websky 1LPT at nside=1024:
    #   export LPT_DISPLACEMENTS_PATH=/pscratch/sd/m/malvarez/websky-displacements/
    #   srun -n $nproc --gpus-per-task=1 xgfield --N 6144 --Nside 1024 --input lptfiles

    import xgfield
    import xgfield.defaults as fmd

    import sys
    import argparse
    from xgfield import fieldsky

    parser = argparse.ArgumentParser(description='Commandline interface to fieldmock')

    parser.add_argument('modelID', type=str)
    parser.add_argument('--N',     type=int, help=f'grid dimension [default = {fmd.N}]',     default=fmd.N)
    parser.add_argument('--Lbox',  type=int, help=f'box size in Mpc [default = {fmd.Lbox}]', default=fmd.Lbox)
    parser.add_argument('--Nside', type=int, help=f'healpix Nside [default = {fmd.Nside}]',  default=fmd.Nside)
    parser.add_argument('--input', type=str, help=f'field input [default = {fmd.input}]',    default=fmd.input)
    args = parser.parse_args()

    ID    = args.modelID
    N     = args.N
    Lbox  = args.Lbox
    Nside = args.Nside
    input = args.input

    sky = fieldsky.FieldSky(ID=ID,N=N,Lbox=Lbox,Nside=Nside,input=input)

    return sky.generate()

if __name__ == "__main__":
    import sys
    sys.exit(main())