import numpy as np
import os
import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import shapely.wkb
import shapely

from scipy.ndimage import binary_erosion
from argparse import ArgumentParser
from subprocess import run
from time import perf_counter
from copy import deepcopy
from operator import countOf
from warnings import filterwarnings
from shutil import rmtree, copyfileobj
from itertools import count
from matplotlib import ticker
from MDAnalysis import Universe
from collections import defaultdict
from sys import exit
from re import split

filterwarnings('ignore', category=UserWarning)
filterwarnings('ignore', category=RuntimeWarning)

parser = ArgumentParser()

# INPUT
parser.add_argument('-traj', '--trajectory', dest='traj', nargs='+',
                    help='MD trajectory-file name in the MD-output-directory. '\
                    'Format is also used for output-files.', type=str)
    
parser.add_argument('-topo', '--topology', dest='topo', default=None,
                    help='MD topology-file name in the MD-output-directory, '\
                    'if trajectory-file does not specify topology. DEFAULT: None.', type=str)
    
parser.add_argument('-fes', '--fes', dest='fes', default=None,
                    help='FES-file name in the MD-output-directory. DEFAULT: None', type=str)

parser.add_argument('-cv', '--colvar', dest='colvar', default='COLVAR', required=True, nargs='+',
                    help='COLVAR-file in the MD-output-directory. DEFAULT: "COLVAR".', type=str)

parser.add_argument('-col, --column', dest='column', default=None, type=str,
                    help='Columns for CV1,CV2,BIAS,RCT (only colvar+reweighting), or CV1,CV2 (only colvar), '\
                        'or CV1(colvar),CV2(colvar),CV1(fes),CV2(fes),Energy (colvar+fes). ' \
                            'Delimited by commas. First column is denoted 1. DEFAULT: 1,2,1,2,3')

# TUNING
parser.add_argument('-thr', '--thresh', dest='thresh', default=None,
                    help='Specifies threshold for assigning. Input value must correspond to values in FES-file. '\
                    'DEFAULT: Lowest 1/12 of the energy span.', type=float)
    
parser.add_argument('-pbc', '--pbc', dest='pbc', action='store_true',
                    help='Use this flag if the FES is periodic.')
    
parser.add_argument('-mind', '--mindist', dest='mindist', default=None,
                    help='Smallest allowed distance at which areas are considered separate minima (in CV units). '\
                    'DEFAULT: 2%% of FES diagonal.', type=float)

parser.add_argument('-thr_l','--thresh_low', dest='thresh_low', default=None, 
                    help='Omit minima with with energies lower than this value. Useful when analyzing '\
                        'highly metastable or transition states.', type=float)

# OPTIONAL
parser.add_argument('-str', '--stride', dest='stride', default=1,
                    help='Reads only every n-th frame of trajectory. If 0 then only outputs a single frame per '\
                        'minimum. DEFAULT: 1.', type=int)

parser.add_argument('-png', '--png', dest='fes_png', default='True', choices=('True','False','only'),
                    help='Specifies whether a PNG-visualization of the FES should be created (True/False) '\
                    'or if only a PNG is desired (only). DEFAULT: True.', type=str)
    
parser.add_argument('-dim', '--dim', dest='dims', default='500,500',
                    help='x-dimension,y-dimension of generated FES if no FES-file provided. DEFAULT: 500,500',
                    type=str)

parser.add_argument('-md', '--md', dest='md_dir', default=os.getcwd(),
                    help='Working path. DEFAULT: Current directory path.', type=str)

parser.add_argument('-kbt', '--kBT', dest='kbt', default=None, type=float,
                    help='Boltzmann-constant * Temperature, only needed if you are doing reweighting')

args = parser.parse_args()


def init_polygon(_singl,_tol):
    global single,tol
    single,tol = _singl,_tol


def init_custom_writer(_fmt,_skoff,_cstraj,_traj):
    global format, seek_offset, cumsum_traj, trajl
    format, seek_offset, cumsum_traj, trajl = _fmt, _skoff, _cstraj, _traj


def have_common_elem(l1, l2):
    for elem in l2:
        if countOf(l1, elem) > 0:
            return True
    return False


def format2lookstr(format):
    if format == 'cfg':
        lookstr = 'BEGIN_CFG'
    elif format == 'lammpstrj':
        lookstr = 'ITEM: TIMESTEP'
    elif format == 'pdb':
        lookstr = 'CRYST1'
    return lookstr
    

def process_chunk(task_data):
    indices, poly_wkb, a, b = task_data
    polygon = shapely.wkb.loads(poly_wkb)
    
    points_geom = shapely.points(a, b)
    dists = shapely.distance(polygon, points_geom)
    mask = dists <= tol
    valid_indices = indices[mask]
    if single and np.any(mask):
        distance_list = shapely.distance(polygon.centroid, points_geom[mask])
        return [valid_indices[np.argmin(distance_list)]]
    else:
        return valid_indices

    
def pos_polygon(parameters, pts):
    new_grid_dimX = int(parameters[8]/(4*np.sqrt(parameters[2]**2+parameters[3]**2)))
    bin_cutoffX = parameters[8]/new_grid_dimX
    bin_cutoffY = parameters[9]/new_grid_dimX    
    center_ids = ((pts[:,0]-parameters[4])/bin_cutoffX).astype(np.int32)+((pts[:,1]-parameters[5])/bin_cutoffY).astype(np.int32)*new_grid_dimX
    
    offsets = np.array([0, -1, 1, new_grid_dimX, -new_grid_dimX, 
                        new_grid_dimX+1, -new_grid_dimX-1, new_grid_dimX-1, -new_grid_dimX+1], dtype=np.int32)
    
    all_ids = (center_ids[:,None]+offsets).ravel()
    return set(all_ids)


def fes_gen_histo(a,b,bias,rct,kBT):
    start_temp = perf_counter()
    dimX, dimY = int(args.dims.split(',')[0]), int(args.dims.split(',')[1])

    if np.any(bias):
        print('generating reweighted histogram ... ', end='', flush=True)
        histo, xedges, yedges = np.histogram2d(a, b, bins=(dimX,dimY), 
                                              range=((min(a),max(a)),(min(b),max(b))), weights=np.exp((bias-rct)/kBT))
    else:
        print('generating normal histogram ... ', end='', flush=True)
        histo, xedges, yedges = np.histogram2d(a, b, bins=(dimX,dimY), 
                                              range=((min(a),max(a)),(min(b),max(b))), density=True)
    xcenters = (xedges[:-1]+xedges[1:])/2
    ycenters = (yedges[:-1]+yedges[1:])/2
    tolX = abs(xcenters[0]-xcenters[1])/2
    tolY = abs(ycenters[0]-ycenters[1])/2
    max_a, max_b = max(xcenters), max(ycenters)
    min_a, min_b = min(xcenters), min(ycenters)
    fullX = abs(max_a - min_a)
    fullY = abs(max_b - min_b)
    coords = np.stack(np.meshgrid(xcenters, ycenters, indexing='ij'), axis=-1)
    coords = np.flipud(np.swapaxes(coords, 0, 1))

    histo[histo == 0] = np.nan
    if kBT is None:
        kBT = 1
    ener2d = np.flipud(-kBT*np.log(histo.T))
    print(f'done in {round(perf_counter() - start_temp,2)} s')
    return (dimX, dimY, tolX, tolY, min_a, min_b, max_a, max_b, fullX, fullY), ener2d, coords


def fes_gen_fes(pos_cvs_fes, pos_ener):
    a_fes, b_fes, ener = np.loadtxt(args.fes, unpack=True, usecols=(*pos_cvs_fes, pos_ener), dtype=float)
    dimX, dimY, ct2 = 0, 1, 0
    count1 = count(0)
        
    if (b_fes[0] == b_fes[1]):
        while b_fes[next(count1)] == b_fes[0]:
            dimX += 1
        b_count = b_fes[0]
        for elem in b_fes:
            if not elem == b_count:
                dimY += 1
                b_count = elem
        max_a, max_b  = a_fes[dimX-1], b_fes[-1]
        tolX = abs(a_fes[0]-a_fes[1])/2
        tolY = abs(b_fes[0]-b_fes[dimY])/2
        ener2d = np.empty((dimY,dimX))
        coords = np.empty((dimY,dimX,2))
        for i in range(dimY):
            for j in range(dimX):
                ener2d[-1-i,j] = ener[ct2]
                coords[-1-i,j,:] = np.array([a_fes[ct2], b_fes[ct2]])
                ct2 += 1
    else:
        while a_fes[next(count1)] == a_fes[0]:
            dimX += 1
        a_count = a_fes[0]
        for elem in a_fes:
            if not elem == a_count:
                dimY += 1
                a_count = elem
        max_a, max_b  = a_fes[-1], b_fes[dimX-1]
        tolX = abs(a_fes[0]-a_fes[dimY])/2
        tolY = abs(b_fes[0]-b_fes[1])/2
        ener2d = np.empty((dimY,dimX))
        coords = np.empty((dimY,dimX,2))
        for i in range(dimX):
            for j in range(dimY):
                ener2d[-1-j,i] = ener[ct2]
                coords[-1-j,i,:] = np.array([a_fes[ct2], b_fes[ct2]])
                ct2 += 1
    min_a, min_b = a_fes[0], b_fes[0]
    fullX = abs(max_a - min_a)
    fullY = abs(max_b - min_b)
    return (dimX, dimY, tolX, tolY, min_a, min_b, max_a, max_b, fullX, fullY), ener2d, coords


def hash_fes(parameters, outline, bins, max_diff):
    bin_cutoffX = np.ceil(max_diff/(2*parameters[2]))    
    bin_cutoffY = np.ceil(max_diff/(2*parameters[3]))
    new_grid_dimX = np.ceil(parameters[0]/bin_cutoffX)
    hash_tab = {}
    for i,b in enumerate(bins):
        posX = np.ceil((b-int(b/parameters[0])*parameters[0])/bin_cutoffX)
        if posX == 0:
            posX = new_grid_dimX
        pos = posX + new_grid_dimX*np.ceil(int(b/parameters[0])/bin_cutoffY)
        try:
            hash_tab[pos].append(outline[i])
        except KeyError:
            hash_tab[pos] = [outline[i]]
    return hash_tab, new_grid_dimX


def hash_colv(parameters,x,y):
    new_grid_dimX = int(parameters[8]/(4*np.sqrt(parameters[2]**2+parameters[3]**2)))
    bin_cutoffX = parameters[8]/new_grid_dimX
    bin_cutoffY = parameters[9]/new_grid_dimX

    hash_tab = defaultdict(list)
    cell_ids = ((x-parameters[4])/bin_cutoffX).astype(np.int64) + ((y-parameters[5])/bin_cutoffY).astype(np.int64)* new_grid_dimX

    for i in tqdm.tqdm(range(len(cell_ids)), desc='filling hashmap', leave=False):
        hash_tab[cell_ids[i]].append(i)
    return hash_tab


def ex3(hash_tab, new_grid_dimX, max_diff):
    separate_groups, subgroup = [], []
    seed_elem_key = list(hash_tab.keys())[0]
    seed_elem_pos = 0
    seed_elem = hash_tab[seed_elem_key][seed_elem_pos]
    collect_bins = set()
    groups_collect_bins = []
    while any(hash_tab):
        min_distance = max_diff
        found = False
        try:
            if not len(hash_tab[seed_elem_key]) == 1:
                del hash_tab[seed_elem_key][seed_elem_pos]
            else:
                del hash_tab[seed_elem_key]
            new_group_found = False
        except KeyError:
            pass

        rel_grid_bins = {seed_elem_key, seed_elem_key+1, seed_elem_key-1, seed_elem_key+new_grid_dimX, 
                         seed_elem_key-new_grid_dimX, seed_elem_key+new_grid_dimX+1, seed_elem_key+new_grid_dimX-1, 
                         seed_elem_key-new_grid_dimX+1, seed_elem_key-new_grid_dimX-1}
        for rel_bin in rel_grid_bins:
            try:
                for i,compare_elem in enumerate(hash_tab[rel_bin]):
                    if ((seed_elem[0]-compare_elem[0])**2 + (seed_elem[1]-compare_elem[1])**2)**0.5 < min_distance: 
                        found = True
                        min_distance = ((seed_elem[0]-compare_elem[0])**2 + (seed_elem[1]-compare_elem[1])**2)**0.5
                        min_elem_key = rel_bin
                        min_elem_pos = i
            except KeyError:
                pass
        if found == True and len(subgroup) > 0:
            collect_bins.update(rel_grid_bins)
            if new_group_found == False:
                subgroup.append(seed_elem)
            seed_elem = hash_tab[min_elem_key][min_elem_pos]
            seed_elem_key = min_elem_key
            seed_elem_pos = min_elem_pos
        else:
            if len(subgroup) > 0:  
                separate_groups.append(subgroup)
            subgroup = []
            sec_run = False
            for k,group in enumerate(separate_groups):
                dists = np.empty(len(group))
                for i,elem in enumerate(group):
                    dists[i] = ((seed_elem[0]-elem[0])**2 + (seed_elem[1]-elem[1])**2)**0.5
                    if dists[i] <= max_diff:
                        sec_run = True
                if sec_run == True:
                    groups_collect_bins[k].update(rel_grid_bins)
                    indx_save = k
                    try:
                        nih = np.empty(len(dists)-1)
                        for j in range(len(dists)-1):
                                nih[j] = dists[j]+dists[j+1]
                        group.insert(np.argmin(nih)+1, seed_elem)
                    except ValueError:
                        group.append(seed_elem)
                    break
            if sec_run == False:
                subgroup.append(seed_elem)
                new_group_found = True
                groups_collect_bins.append(collect_bins)
                collect_bins = set()
            elif any(hash_tab):
                seed_elem_pos = 0
                curr_key_list = set(hash_tab.keys())
                if any(curr_key_list.intersection(collect_bins)):
                    seed_elem_key = list(curr_key_list.intersection(collect_bins))[0]
                elif any(groups_collect_bins[indx_save].intersection(curr_key_list)):
                    seed_elem_key = list(groups_collect_bins[indx_save].intersection(curr_key_list))[0]
                else:
                    seed_elem_key = list(hash_tab.keys())[0]
                seed_elem = hash_tab[seed_elem_key][seed_elem_pos]

    connect_groups = []      
    for g1 in range(0,len(separate_groups)):
        for g2 in range(g1+1, len(separate_groups)):
            if groups_collect_bins[g1]&groups_collect_bins[g2]:
                dists = np.empty(len(separate_groups[g1])*len(separate_groups[g2]))
                indx = count(0)
                for e1 in separate_groups[g1]:
                    for e2 in separate_groups[g2]:
                        dists[next(indx)] = (((e1[0]-e2[0])**2+(e1[1]-e2[1])**2)**0.5)
                if np.min(dists) <= max_diff :
                    connect_groups.append([g1,g2])

    grouped_connected_groups = []
    while len(connect_groups)>0:
        first, *rest = connect_groups
        first = set(first)
        lf = -1
        while len(first)>lf:
            lf = len(first)
            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2
        grouped_connected_groups.append(first)
        connect_groups = rest
        
    fin_sep_groups, tot  = [], []
    for elem in grouped_connected_groups:
        help_list = []
        for i in elem:
            tot.append(i)
            help_list += separate_groups[i]
        fin_sep_groups.append(help_list)
    
    for i,elem in enumerate(separate_groups):
        if not i in tot:
            fin_sep_groups.append(elem)
    return fin_sep_groups


def printout_custom(data):
    workeridx,curr_indx = data
    current_file_idx = -1
    ftraj = None
    currfile = f'minima/temp_{workeridx}.'+format
    lookstr = format2lookstr(format).encode()
    try:
        with open(currfile, 'wb') as minfile:
            file_indices = np.searchsorted(cumsum_traj, curr_indx, side='right')
            for j,idx in enumerate(curr_indx):
                file_seek_idx = int(idx-cumsum_traj[file_indices[j]-1]) if file_indices[j] > 0 else idx
                if not file_indices[j] == current_file_idx:
                    if ftraj:
                        ftraj.close()
                    ftraj = open(trajl[file_indices[j]], 'rb')
                    current_file_idx = file_indices[j]

                ftraj.seek(seek_offset[file_indices[j]][file_seek_idx])
                first_line = ftraj.readline()
                minfile.write(first_line)
                for line_bytes in ftraj:
                    if line_bytes.startswith(lookstr):
                        break
                    minfile.write(line_bytes)
    finally:
        if ftraj:
            ftraj.close()
    return currfile


def printout_prework(end, trajl):
    lookstr = format2lookstr(end)
    file_offset = []
    traj_len = np.empty(len(trajl), dtype=np.uint64)
    try: #UNIX
        for i,filename in enumerate(trajl):
            result = run(f"grep -b '^{lookstr}' {filename} | cut -d: -f1", shell=True, capture_output=True, text=True)
            seek_offset = np.fromstring(result.stdout, dtype=int, sep='\n')
            file_offset.append(seek_offset)
            traj_len[i] = len(seek_offset)
    except FileNotFoundError: #WINDOWS
        lookstr = lookstr.encode()
        for i,filename in enumerate(trajl):
            seek_offset = []
            with open(filename,'rb') as f:
                line_offset = 0
                for line in f:
                    if line.startswith(lookstr):
                        seek_offset.append(line_offset)
                    line_offset += len(line)
            file_offset.append(seek_offset)
            traj_len[i] = len(seek_offset)
    return file_offset, traj_len


def stdout(string, center=False, end='\n', start=''):
    try:
        terminal_size = os.get_terminal_size()[0]
    except OSError:
        terminal_size = len(string)
    if not center:
        print(start+string, end=end, flush=True)
    else:
        tempstr = ' '*int((terminal_size-len(string+start))/2)
        print(start+tempstr+string+tempstr, flush=True)


if __name__ == '__main__':
    start = perf_counter()
    title = '.: Free Energy Surface Trajectory Analysis - FESTA :.'
    termin = '.: terminated successfully :.'
    version = '.: standalone version :.'

    stdout(title, center=True, start='\n')
    stdout(version, center=True)
    stdout(f'working directory: {args.md_dir}', start='\n')
    os.chdir(args.md_dir)
    args.colvar = sorted(args.colvar, key=lambda el:[int(c) if c.isdigit() else c for c in split(r'(\d+)', el)])


    if args.fes is None:
        if args.column is None:
            pos_cvs_colv = (0,1)
        else:
            pos_cvs_colv = [int(pos)-1 for pos in args.column.split(',')]
        assert len(pos_cvs_colv) in (2,4), "Expects either 2 (no reweighting) or 4 (reweighting) COLVAR columns when no FES given"
    else:
        if args.column is None:
            pos_cvs_colv = (0,1)
            pos_cvs_fes = (0,1)
            pos_ener = 2
        else:
            pos_cvs_pos = [int(pos)-1 for pos in args.column.split(',')]
            assert len(pos_cvs_pos) == 5, "Expects 5 columns (CV1(colvar),CV2(colvar),CV1(fes),CV2(fes),Energy)"
            pos_cvs_fes = pos_cvs_pos[2:4]
            pos_ener = pos_cvs_pos[4]
            pos_cvs_colv = pos_cvs_pos[:2]

    start2 = perf_counter()
    print('reading colvar file ... ' , end='', flush=True)
    a, b, bias, rct = [], [], [], []
    if len(pos_cvs_colv) == 2:
        for element in args.colvar:
            try:
                a_tmp, b_tmp = np.loadtxt(element, unpack=True, usecols=pos_cvs_colv, dtype=float)
            except UnicodeDecodeError:
                tmp = np.load(element)
                a_tmp, b_tmp = tmp[:,0],tmp[:,1]
            a.append(a_tmp)
            b.append(b_tmp)
    else:
        assert args.kbt is not None, 'kBT must be given when doing reweighting'
        for element in args.colvar:
            try:
                a_tmp, b_tmp, bias_tmp, rct_tmp = np.loadtxt(element, unpack=True, usecols=pos_cvs_colv, dtype=float)
            except UnicodeDecodeError:
                tmp = np.load(element)
                a_tmp, b_tmp, bias, rct = tmp[:,0],tmp[:,1],tmp[:,2],tmp[:,3]
            a.append(a_tmp)
            b.append(b_tmp)
            bias.append(bias_tmp)
            rct.append(rct_tmp)
        bias = np.concatenate(bias)
        rct = np.concatenate(rct)
    
    a = np.concatenate(a)
    b = np.concatenate(b)
    print(f'done in {round(perf_counter() - start2,2)} s')

    single = False
    if args.stride > 1:
        a, b, bias, rct = a[::args.stride], b[::args.stride], bias[::args.stride], rct[::args.stride]
        print(f'applied {args.stride} stride')
    elif args.stride == 0:
        args.stride = 1
        single = True

    parameters, ener2d, coords = fes_gen_fes(pos_cvs_fes, pos_ener) if args.fes else fes_gen_histo(a,b,bias,rct,args.kbt)

    if args.thresh is None:
        max_ener = np.nanmax(ener2d)
        args.thresh = max_ener - abs(max_ener-np.nanmin(ener2d))*(1-1/12)
        stdout('automatically determined', end=' ') 
    stdout(f'threshold value: {round(args.thresh,3)} a.U.')

    mask_valid = ener2d < args.thresh  
    rows, cols = parameters[1], parameters[0]
    mask_grid_edge = np.zeros_like(mask_valid, dtype=bool)
    mask_grid_edge[0,:] = True
    mask_grid_edge[-1,:] = True
    mask_grid_edge[:,0] = True
    mask_grid_edge[:,-1] = True

    mask_eroded = binary_erosion(mask_valid, structure=np.ones((3,3)), border_value=0)
    mask_outline = mask_valid & (~mask_eroded)
    pol_bins = coords[mask_valid]
    ener_bins = ener2d[mask_valid]

    outline = coords[mask_outline]
    edge = coords[mask_valid & mask_grid_edge]

    flat_indices = np.arange(1, rows*cols+1).reshape(rows, cols)
    bins = flat_indices[mask_outline]
    
    if args.mindist == None:
        args.mindist = np.sqrt((parameters[6]-parameters[4])**2+(parameters[7]-parameters[5])**2)*0.02
    elif args.mindist < 2*np.sqrt(parameters[2]**2+parameters[3]**2):
        raise Exception('Minimal separation distance must be larger than '\
                        f'diagonal of a single bin:{2*np.sqrt(parameters[2]**2+parameters[3]**2)} a.U.')
    print(f'separation distance: {round(args.mindist,5)} a.U.')

    if not args.fes_png == 'only':
        format = args.traj[0].split('.')[-1]
        print('reading trajectory ... ' , end='', flush=True)
        args.traj = sorted(args.traj, key=lambda el:[int(c) if c.isdigit() else c for c in split(r'(\d+)', el)])
        try:
            if args.topo == None:
                u = Universe(args.traj[0], args.traj)
            else:
                u = Universe(args.topo, args.traj, atom_style='atomic')
            ag = u.select_atoms('all')
            use_MDAnalysis = True
            if not int((len(u.trajectory)-1)/args.stride+1) == len(a):
                raise Exception('COLVAR-file and trajectory-file must have similar step length,'/
                                f' here: {len(a)} vs {int((len(u.trajectory)-1)/args.stride+1)}')
        except (IndexError, ValueError):
            use_MDAnalysis = False
            assert format in ('cfg','lammpstrj','pdb'), 'Format not supported by MDAnalysis or CustomWriter'
            seek_offset, traj_len = printout_prework(format, args.traj)
            frame_count = sum(traj_len)
            cumsum_traj = np.cumsum(traj_len)
            if not int((frame_count-1)/args.stride+1) == len(a):
                raise Exception('COLVAR-file and trajectory-file must have similar step length,'/
                                f' here: {len(a)} vs {int((frame_count-1)/args.stride+1)}')
        print('done')

        all_points = hash_colv(parameters, a, b)
    
    print('executing CCL step ... ', end='', flush=True)
    start0 = perf_counter()
    hash_list, new_dimX = hash_fes(parameters, outline, bins, args.mindist)

    grouped_points = ex3(hash_list, new_dimX, args.mindist)
    grouped_points = [groups for groups in grouped_points if len(groups)>3]
    print(f'done in {round(perf_counter() - start0,2)} s')
    
    start1 = perf_counter()
    periodicity = False
    if edge.size > 0 and args.pbc:
        edge_points, pbc = [], []
        grouped_edges = ex3(edge, 10*2*np.sqrt(parameters[2]**2+parameters[3]**2))
        for i in range(len(grouped_edges)):
            if sum(list(map(len, pbc))) >= len(grouped_edges):
                break
            expect_group, tmp_lst = [], []
            for elem in grouped_edges[i]:
                tmp_pt = deepcopy(elem)
                if elem[0] == parameters[6]:
                    tmp_pt[0] = parameters[4]
                elif elem[0] == parameters[4]:
                    tmp_pt[0] = parameters[6]
                if elem[1] == parameters[7]:
                    tmp_pt[1] = parameters[5]
                elif elem[1] == parameters[5]:
                    tmp_pt[1] = parameters[7]
                expect_group.append(tmp_pt)
            found_periodic = False
            for j,group2 in enumerate(grouped_points):
                if have_common_elem(group2, expect_group) or have_common_elem(group2, grouped_edges[i]):
                    periodicity = True
                    found_periodic = True
                    tmp_lst.append(j)
            if found_periodic == True:
                if len(tmp_lst) == 1:
                    break
                elif i == 0:
                    stdout('periodicity detected: boundaries will be considered periodic')
                pbc.append(tmp_lst)
    print(str(len(grouped_points)), end = ' ')
    if periodicity or args.thresh_low:
        print('distinctive areas identified')
    else:
        print('minimum identified') if len(grouped_points) == 1 else print('minima identified')
    
    tot_min_frames = 0
    roids = []
        
    usable_cpu = os.cpu_count()
    sorted_indx, exteriors_x, exteriors_y = [], [], []
    tol = np.sqrt(parameters[3]**2 + parameters[2]**2)
    with mp.Pool(processes=usable_cpu, initializer=init_polygon, initargs=(single,tol), maxtasksperchild=100) as pool:
        for j in tqdm.tqdm(range(len(grouped_points)), desc='polygon distance', leave=False):
            polygon = shapely.Polygon(grouped_points[j]).buffer(0)
            pol_fill = []
            skip_polygon = False
            for k,fes_bin in enumerate(pol_bins):
                if polygon.distance(shapely.Point(fes_bin)) <= args.mindist:
                    if args.thresh_low is None or args.thresh_low < ener_bins[k]:
                        pol_fill.append(fes_bin)
                    else:
                        skip_polygon = True
                        break
            if skip_polygon:
                continue
            pol_fill = np.array(pol_fill)

            try:
                exteriors_x.append(np.abs((polygon.exterior.xy[0]-parameters[4])/
                                          ((parameters[6]-parameters[4])/parameters[0])))
                exteriors_y.append(parameters[1]-np.abs((polygon.exterior.xy[1]-parameters[5])/
                                                        ((parameters[7]-parameters[5])/parameters[1])))
            except AttributeError:
                exteriors_x_tmp, exteriors_y_tmp = [], []
                for poly in polygon.geoms:
                    exteriors_x_tmp += (np.abs((poly.exterior.xy[0]-parameters[4])/
                                               ((parameters[6]-parameters[4])/parameters[0]))).tolist()
                    exteriors_y_tmp += (parameters[1]-np.abs((poly.exterior.xy[1]-parameters[5])/
                                                             ((parameters[7]-parameters[5])/parameters[1]))).tolist()
                exteriors_x.append(exteriors_x_tmp)
                exteriors_y.append(exteriors_y_tmp)

            if args.fes_png == 'only':
                continue

            roids.append(polygon.centroid.coords[0])
            poly_wkb = shapely.wkb.dumps(polygon)
            
            pol_fill_keys = pos_polygon(parameters, pol_fill)
            candidate_indices = []
            for key in pol_fill_keys:
                candidate_indices.extend(all_points.get(key, []))
            
            candidate_indices = np.array(candidate_indices, dtype=np.int64)
            num_chunks = int(np.ceil(len(candidate_indices)/50000))
            num_chunks = max(num_chunks, usable_cpu*4)

            if len(candidate_indices) < 200000:
                final_indices = process_chunk((candidate_indices, poly_wkb, a[candidate_indices], b[candidate_indices]))
            else:
                chunks = np.array_split(candidate_indices, num_chunks)
                tasks = [(chunk, poly_wkb, a[chunk], b[chunk]) for chunk in chunks]
                
                final_indices = []
                for r_idx in pool.imap_unordered(process_chunk, tasks):
                    if len(r_idx) > 0:
                        final_indices.extend(r_idx)

            if single:
                distance_list = shapely.distance(polygon.centroid, shapely.points(a[final_indices], b[final_indices]))
                sorted_indx.append([final_indices[np.argmin(distance_list)]])
            else:
                final_indices.sort()
                sorted_indx.append(final_indices)
            tot_min_frames += len(sorted_indx[-1])
            
    roids = np.array(roids)
    if args.thresh_low:
        skipped_poly = len(grouped_points)-len(exteriors_x)
        print(f'omitted {skipped_poly} area(s) due to energies lower than {args.thresh_low} a.U.')
        print(f'{len(exteriors_x)} minima identified')

    try:
        os.mkdir('minima')
    except FileExistsError:
        rmtree('minima')
        os.mkdir('minima')

    if not args.fes_png == 'False':
        plt.figure(figsize=(8,6), dpi=300)
        plt.imshow(ener2d, interpolation='none', cmap='nipy_spectral')
        plt.xticks(np.linspace(-0.5,parameters[0]-0.5,5),np.round(np.linspace(parameters[4],parameters[6],5),3))
        plt.yticks(np.linspace(-0.5,parameters[1]-0.5,5),np.round(np.linspace(parameters[7],parameters[5],5),3))
        plt.xlabel('CV1 [a.U.]')
        plt.ylabel('CV2 [a.U.]')
        plt.axis('tight')
        plt.title(f'threshold: {round(args.thresh,3)} a.U.')
        neon_colors = ['#39FF14','#FF00FF','#00FFFF','#FFFF00', 
            '#FF3131','#FF9933','#B026FF']
        
        for i in range(len(exteriors_x)):   
            plt.plot(exteriors_x[i], exteriors_y[i], '-', color=neon_colors[i%len(neon_colors)], 
                linewidth=1, path_effects=[pe.Stroke(linewidth=2, foreground='white'), pe.Normal()])
            
        cb = plt.colorbar(label='Energy [a.U.]', format="{x:.1f}")
        tick_locator = ticker.MaxNLocator(nbins=8)
        cb.locator = tick_locator
        cb.update_ticks()    
        plt.savefig('minima/fes_visual.png',bbox_inches='tight')
        if args.fes_png == 'only':
            stdout(termin, center=True, start='\n')
            exit()

    stdout(f'processed {len(a)} frames')
    stdout(f'found {tot_min_frames} minima frames')
    if tot_min_frames/len(a) > 0.9:
        stdout(f'WARNING: {round(tot_min_frames/len(a)*100)}% of frames part of minima, check if this really is what you want')
    stdout(f'time needed for minima frames identification step: {round(perf_counter() - start1,3)} s')
    
    desc = []
    if periodicity:
        sorted_coords_period, tot_pbc  = [], []
        for elem in pbc:
            desc.append(' + '.join((f'CV1: {round(roids[j,0],4)} CV2: {round(roids[j,1],4)}') for j in elem))
            help_list = []
            for i in elem:
                tot_pbc.append(i)
                help_list += sorted_indx[i]
            if not use_MDAnalysis:
                help_list.sort()
            sorted_coords_period.append(help_list)
        for i,elem in enumerate(sorted_indx):
            if not i in tot_pbc:
                desc.append(f'CV1: {round(roids[i,0],4)} CV2: {round(roids[i,1],4)}')
                sorted_coords_period.append(elem)
        sorted_indx = sorted_coords_period
        print(str(len(sorted_indx)), end=' ')
        print('minimum identified') if len(sorted_indx) == 1 else print('minima identified')
            
    start3 = perf_counter()
    with open('minima/min_overview.txt', 'w') as overviewfile:
        for i in range(len(sorted_indx)):
            if desc:
                overviewfile.writelines(f'min_{i} : {desc[i]} \n')
            else:
                overviewfile.writelines(f'min_{i} : CV1: {round(roids[i,0],4)} CV2: {round(roids[i,1],4)}\n')
    if use_MDAnalysis:
        for i in tqdm.tqdm(range(len(sorted_indx)), desc='writing frames', leave=False):
            curr_indx = sorted_indx[i]*args.stride
            try:
                ag.write(f'minima/min_{i}.{format}', frames=u.trajectory[curr_indx])
                writer_avail = True
            except (TypeError, ValueError):
                writer_avail = False
                ag.write(f'minima/min_{i}.xyz', frames=u.trajectory[curr_indx])
        if not writer_avail:
            print(f'MDAnalysis does not support writing in {format}-format, frames are in xyz-format instead')
    else:
        if single:
            usable_cpu = 1
        with mp.Pool(processes=usable_cpu, initializer=init_custom_writer, 
                    initargs=(format,seek_offset,cumsum_traj,args.traj)) as pool:
            with tqdm.tqdm(total=len(sorted_indx)*usable_cpu, desc='writing frames', leave=False) as pbar:
                for i in range(len(sorted_indx)):
                    indxsplit = np.array_split(np.array(sorted_indx[i])*args.stride, usable_cpu)
                    with open(f'minima/min_{i}.'+format, 'wb') as outfile:
                        for filename in pool.imap_unordered(printout_custom, enumerate(indxsplit)):
                            pbar.update(1)
                            with open(filename, 'rb') as infile:
                                copyfileobj(infile, outfile)
                            os.remove(filename)

    stdout(f'time needed for postprocessing step: {round(perf_counter() - start3,3)} s')
    stdout(termin, center=True, start='\n')
