"""Module map_functions.
This module includes functions to manipulate maps.

Notes
-----
    KrCalib code depends on the IC library.
    Public functions are documented using numpy style convention

Documentation
-------------
    Insert documentation https
"""


import pandas as pd

from . kr_types       import FitParTS
from . kr_types       import ASectorMap
from . kr_types       import SectorMapTS
from . kr_types       import FitMapValue

from typing           import List
from typing           import Tuple
from typing           import Dict

import logging


log = logging.getLogger(__name__)


def tsmap_from_fmap(fMap : Dict[int, List[FitParTS]])->SectorMapTS:
    """
    Obtain a time-series of maps (tsmap) from a fit-map (fmap).

    Parameters
    ----------
    fMap
        A Dictionary (key = R sector for Rphi maps, X for XYmaps) containing a list of FitParTS
        (list runs over Phi wedges for RPhi maps, Y for Ymaps)
        class ASectorMap:
            chi2  : DataFrame
            e0    : DataFrame
            lt    : DataFrame
            e0u   : DataFrame
            ltu   : DataFrame

            class FitParTS:
                ts   : np.array -> contains the time series (integers expressing time differences)
                e0   : np.array ->e0 fitted in time series
                lt   : np.array
                c2   : np.array
                e0u  : np.array
                ltu  : np.array

    Returns
    -------
    SectorMapTS : Maps in chamber sector containing time series of parameters
        class SectorMapTS:
            chi2  : Dict[int, List[np.array]]
            e0    : Dict[int, List[np.array]]
            lt    : Dict[int, List[np.array]]
            e0u   : Dict[int, List[np.array]]
            ltu   : Dict[int, List[np.array]]

    """
    logging.debug(f' --tsmap_from_fmap')
    tmChi2 = {}
    tmE0   = {}
    tmLT   = {}
    tmE0u  = {}
    tmLTu  = {}

    for sector, fps in fMap.items():
        logging.debug(f' filling maps for sector {sector}')

        tCHI2 = [fp.c2  for fp in fps]
        tE0   = [fp.e0  for fp in fps]
        tLT   = [fp.lt  for fp in fps]
        tE0u  = [fp.e0u for fp in fps]
        tLTu  = [fp.ltu for fp in fps]

        tmChi2[sector]  = tCHI2
        tmE0  [sector]  = tE0
        tmLT  [sector]  = tLT
        tmE0u [sector]  = tE0u
        tmLTu [sector]  = tLTu

    return SectorMapTS(chi2 = tmChi2,
                       e0   = tmE0  ,
                       lt   = tmLT  ,
                       e0u  = tmE0u ,
                       ltu  = tmLTu )


def amap_from_tsmap(tsMap      : SectorMapTS,
                    ts         : int                 =             0,
                    range_e    : Tuple[float, float] = (5000, 13000),
                    range_chi2 : Tuple[float, float] = (   0,     3),
                    range_lt   : Tuple[float, float] = (1800,  3000)) -> ASectorMap:
    """
    Obtain the correction maps for time bin ts.

    Parameters
    ----------
    tsMap
        A SectorMapTS : Maps in chamber sector containing time series of parameters
        class SectorMapTS:
            chi2  : Dict[int, List[np.array]]
            e0    : Dict[int, List[np.array]]
            lt    : Dict[int, List[np.array]]
            e0u   : Dict[int, List[np.array]]
            ltu   : Dict[int, List[np.array]]
    ts
        time bin (an integer starting at 0
    range_e
        Defines the range of e in pes (e.g, (8000,14000)).
    range_chi2
        Defines the range of chi2
    range_lt
        Defines the range of lt in mus.

    Returns
    -------
    A container of maps ASectorMap
        class ASectorMap:
            chi2  : DataFrame
            e0    : DataFrame
            lt    : DataFrame
            e0u   : DataFrame
            ltu   : DataFrame

    """

    def fill_map_ts(tsm : Dict[int, List[float]], ts : int):
        M = {}
        for sector, w in tsm.items():
            M[sector] = [v[ts] for v in w]
        return M

    if ts >= 0:
        mChi2 = fill_map_ts(tsMap.chi2, ts)
        mE0   = fill_map_ts(tsMap.e0  , ts)
        mLT   = fill_map_ts(tsMap.lt  , ts)
        mE0u  = fill_map_ts(tsMap.e0u , ts)
        mLTu  = fill_map_ts(tsMap.ltu , ts)
    return ASectorMap(chi2    = pd.DataFrame.from_dict(mChi2),
                      e0      = pd.DataFrame.from_dict(mE0  ),
                      lt      = pd.DataFrame.from_dict(mLT  ),
                      e0u     = pd.DataFrame.from_dict(mE0u ),
                      ltu     = pd.DataFrame.from_dict(mLTu ),
                      mapinfo = None)


def add_mapinfo(asm        : ASectorMap,
                xr         : Tuple[float, float],
                yr         : Tuple[float, float],
                nx         : int,
                ny         : int,
                run_number : int) -> ASectorMap:
    """
    Add metadata to a ASectorMap

        Parameters
        ----------
            asm
                ASectorMap object.
            xr, yr
                Ranges in (x, y) defining the map.
            nx, ny
                Number of bins in (x, y) defining the map.
            run_number
                run number defining the map.

        Returns
        -------
            A new ASectorMap containing metadata (in the variable mapinfo)

    """
    mapinfo = pd.Series([*xr, *yr, nx, ny, run_number],
                         index=['xmin','xmax',
                                'ymin','ymax',
                                'nx'  , 'ny' ,
                                'run_number'])

    return ASectorMap(chi2    = asm.chi2,
                      e0      = asm.e0  ,
                      lt      = asm.lt  ,
                      e0u     = asm.e0u ,
                      ltu     = asm.ltu ,
                      mapinfo = mapinfo )


def amap_average(amap : ASectorMap) -> FitMapValue:
    return FitMapValue(chi2    = amap.chi2.mean().mean(),
                       e0      = amap.e0  .mean().mean(),
                       lt      = amap.lt  .mean().mean(),
                       e0u     = amap.e0u .mean().mean(),
                       ltu     = amap.ltu .mean().mean())


def amap_max(amap : ASectorMap) -> FitMapValue:
    return FitMapValue(chi2    = amap.chi2.max().max(),
                       e0      = amap.e0  .max().max(),
                       lt      = amap.lt  .max().max(),
                       e0u     = amap.e0u .max().max(),
                       ltu     = amap.ltu .max().max())


def amap_min(amap : ASectorMap) -> FitMapValue:
    return FitMapValue(chi2    = amap.chi2.min().min(),
                       e0      = amap.e0  .min().min(),
                       lt      = amap.lt  .min().min(),
                       e0u     = amap.e0u .min().min(),
                       ltu     = amap.ltu .min().min())


def amap_replace_nan_by_mean(amap : ASectorMap) -> ASectorMap:
    amap_mean = amap_average(amap)
    return ASectorMap(chi2    = amap.chi2.fillna(amap_mean.chi2),
                      e0      = amap.e0  .fillna(amap_mean.e0  ),
                      lt      = amap.lt  .fillna(amap_mean.lt  ),
                      e0u     = amap.e0u .fillna(amap_mean.e0u ),
                      ltu     = amap.ltu .fillna(amap_mean.ltu ),
                      mapinfo = amap.mapinfo)


def amap_replace_nan_by_value(amap : ASectorMap, val : float = 0) -> ASectorMap:
    return ASectorMap(chi2    = amap.chi2.fillna(val),
                      e0      = amap.e0  .fillna(val),
                      lt      = amap.lt  .fillna(val),
                      e0u     = amap.e0u .fillna(val),
                      ltu     = amap.ltu .fillna(val),
                      mapinfo = amap.mapinfo)

def amap_copy(amap : ASectorMap) -> ASectorMap:
    return ASectorMap(chi2  = amap.chi2.copy(),
                      e0    = amap.e0.copy(),
                      lt    = amap.lt.copy(),
                      e0u   = amap.e0u.copy(),
                      ltu   = amap.ltu.copy(),
                      mapinfo   = None)

# n_min is minimum number of neighbors. You can use this
# to say, for example, "don't fill a point that has only
# one filled in neighbor, wait until more of its
# neighbors have been filled"
def fill_neighborhoods(amap_old : ASectorMap , n_min : float = 4) -> ASectorMap:
    amap = asm_copy(amap_old)
    nbins = len(amap.lt)
    for i in range(0, nbins):
        for j in range(0, nbins):
            if math.isnan(amap.lt[i][j]):
                nbhd_mean_lt = 0
                nbhd_mean_ltu = 0
                nbhd_mean_e0 = 0
                nbhd_mean_e0u = 0
                n_nbhs = 0
                for di in range(max(i-1,0),min(nbins,i+2)):
                    for dj in range(max(j-1,0),min(nbins,j+2)):
                        if not math.isnan(amap.lt[di][dj]):
                            nbhd_mean_lt += amap.lt[di][dj]
                            nbhd_mean_ltu += amap.ltu[di][dj]
                            nbhd_mean_e0 += amap.e0[di][dj]
                            nbhd_mean_e0u += amap.e0u[di][dj]
                            n_nbhs += 1
                if n_nbhs > n_min:
                    nbhd_mean_lt /= n_nbhs
                    nbhd_mean_ltu /= n_nbhs
                    nbhd_mean_e0 /= n_nbhs
                    nbhd_mean_e0u /= n_nbhs
                    amap.lt[i][j] = nbhd_mean_lt
                    amap.ltu[i][j] = nbhd_mean_ltu
                    amap.e0[i][j] = nbhd_mean_e0
                    amap.e0u[i][j] = nbhd_mean_e0u

    return amap

# Make sure everyone who has at least 5 neighbors is filled,
# then iterate until you're just left with clusters of points
# that all have less than 5 neighbors
def iterate_fill_neighborhoods(amap_old : ASectorMap , n_min : float = 4) -> ASectorMap:
    
    amap_new = amap_copy(amap_old)
    amap_replaced = fill_neighborhoods(amap_new, n_min)
    previous_lts = np.array(amap_replace_nan_by_value(amap_new).lt)
    replaced_lts = np.array(amap_replace_nan_by_value(amap_replaced).lt)
    
    # start by iteratively filling those with > n_min neighbors
    while not np.allclose(previous_lts, replaced_lts):
        amap_replaced = fill_neighborhoods(amap_replaced, n_min)
        previous_lts = replaced_lts.copy()
        replaced_lts = np.array(amap_replace_nan_by_value(amap_replaced).lt)
    
    return amap_replaced

# Check if amap has values in every spot
# in the lifetime map. Return false if
# any nans remain.
def amap_full(amap):
    for i in range(len(amap.lt)):
        for j in range(len(amap.lt[i])):
            if np.isnan(amap.lt[i][j]):
                return False
    return True

# Start with iterate_fill_neighborhoods, then do the same
# for n_min = 4, then 3, then 2, then 1. Repeat the process
# over until the entire map is full.
def fill_neighborhoods_descending(amap_old : ASectorMap) -> ASectorMap:

    num_cycles = 0
    amap_new = amap_copy(amap_old)
    
    while not amap_full(amap_new) and num_cycles < 10:
        num_cycles += 1
        
        for min in range(0, 6):
            amap_new = fill_neighborhoods(amap_new, n_min = (5 - min))
    
    return amap_new

def amap_replace_nan_by_nbhd(amap : ASectorMap) -> ASectorMap:
    return ASectorMap(chi2    = amap.chi2.fillna(val),
                      e0      = amap.e0  .fillna(val),
                      lt      = amap.lt  .fillna(val),
                      e0u     = amap.e0u .fillna(val),
                      ltu     = amap.ltu .fillna(val),
                      mapinfo = amap.mapinfo)


def relative_errors(am : ASectorMap) -> ASectorMap:
    return ASectorMap(chi2    = am.chi2,
                      e0      = am.e0,
                      lt      = am.lt,
                      e0u     = 100 * am.e0u / am.e0,
                      ltu     = 100 * am.ltu / am.lt,
                      mapinfo = am.mapinfo)
