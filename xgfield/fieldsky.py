def _get_websky_displacements():
    # Websky displacement fields
    try:
        path2disp = os.environ['LPT_DISPLACEMENTS_PATH']
    except:
        path2disp = '/Users/shamik/Documents/Work/websky_datacube/'

    backend.print2log(log, f"Path to displacement files set to {path2disp}", level='usky_info')

    if grid_nside == 768:
        sxfile = path2disp+'sx1_7700Mpc_n6144_nb30_nt16_no768'
        syfile = path2disp+'sy1_7700Mpc_n6144_nb30_nt16_no768'
        szfile = path2disp+'sz1_7700Mpc_n6144_nb30_nt16_no768'
    else:
        sxfile = path2disp+'sx1_7700Mpc_n6144_nb30_nt16'
        syfile = path2disp+'sy1_7700Mpc_n6144_nb30_nt16'
        szfile = path2disp+'sz1_7700Mpc_n6144_nb30_nt16'

    displacements = {}
    displacements['type'] = 'filelist'
    displacements['data'] = [sxfile, syfile, szfile]

    return displacements

class FieldSky:
    '''FieldSky'''
    def __init__(self, **kwargs):

        import xgfield.defaults as fd
        self.ID    = kwargs.get(    'ID',fd.ID)
        self.N     = kwargs.get(     'N',fd.N)
        self.Lbox  = kwargs.get(  'Lbox',fd.Lbox)
        self.Nside = kwargs.get( 'Nside',fd.Nside)
        self.input = kwargs.get( 'input',fd.input)
        self.nproc = kwargs.get( 'nproc',1)
        self.rank  = kwargs.get(  'rank',0)

        self.cube  = kwargs.get(  'cube')
        if self.cube is not None:
            self.input = 'cube'
            self.N    = self.cube.N
            self.Lbox = self.cube.Lbox

        if self.input == 'websky':
            self.displacements = _get_websky_displacements()
        elif self.input == 'cube':
            self.displacements = {}
            self.displacements['type']  = 'arraylist'
            self.displacements['data']  = [cube.s1x,cube.s1y,cube.s1z,cube.s2x,cube.s2y,cube.s2z]
            self.displacements['start'] = (     0,self.N//nproc*rank,         0)
            self.displacements['end']   = (self.N,self.N//nproc*(rank+1),self.N)

         return

    def generate(self, **kwargs):

        from time import time
        times={'t0' : time()}

        import os
        import argparse
        import logging
        import xgfield.libfield as lfm
        import xgfield.defaults as fmd
        import xgcosmo.cosmology   as cosmo
        import xgutil.backend      as bk

        log = logging.getLogger("LIGHTCONE")

        grid_nside = self.N # cube shape is parameterized by grid_nside; full resolution for websky is 6144
        map_nside  = self.Nside
        L_box      = self.Lbox

        # ------ hardcoded parameters
        comov_lastscatter_Gpc = 13.8  # conformal distance to last scattering surface in Gpc
        zmin                  = 0.05  # minimum redshift for projection (=0.05 for websky products)
        zmax                  = 4.5   # maximum redshift for projection (=4.50 for websky products)

        force_no_mpi          = False 
        force_no_gpu          = False

        kappa_map_filebase = f'./output/kappa_1lpt_grid-{ grid_nside }_nside-{ map_nside }'

        backend = bk.Backend(force_no_mpi=force_no_mpi, force_no_gpu=force_no_gpu,logging_level=-logging.ERROR)
        backend.print2log(log, f"Backend configuration complete.", level='usky_info')

        backend.print2log(log, f"Computing cosmology...", level='usky_info')
        cosmo_wsp = cosmo.cosmology(backend, Omega_m=0.31, h=0.68) # for background expansion consistent with websky
        backend.print2log(log, f"Cosmology computed", level='usky_info')

        backend.print2log(log, f"Setting up lightcone workspace...", level='usky_info')
        lpt_wsp = lfm.LibField(cosmo_wsp, grid_nside, map_nside, L_box, zmin, zmax)

        backend.print2log(log, f"Computing LPT to kappa map...", level='usky_info')
        kappa_map = lpt_wsp.fieldmap(self.displacements, backend, bytes_per_cell=4)
        backend.print2log(log, f"Kappa map computed. Saving to file.", level='usky_info')

        backend.mpi_backend.writemap2file(kappa_map, kappa_map_filebase+".fits")
        backend.print2log(log, f"LIGHTCONE: Kappa map saved. Exiting...", level='usky_info')

        return 0




