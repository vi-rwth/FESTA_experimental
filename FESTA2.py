import numpy as np
import os
import shapely
import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import argparse

from subprocess import run
from time import perf_counter
from copy import deepcopy
from operator import countOf
from warnings import filterwarnings
from shutil import rmtree
from itertools import count
from matplotlib import ticker
from MDAnalysis import Universe
from collections import defaultdict
from sys import exit

filterwarnings('ignore', category=UserWarning)
filterwarnings('ignore', category=RuntimeWarning)

parser = argparse.ArgumentParser()

# INPUT
parser.add_argument('-traj', '--trajectory', dest='traj', required=True, nargs='+',
                    help='!REQUIRED! MD trajectory-file name in the MD-output-directory. '\
                    'Format is also used for output-files.',
                    type=str)
    
parser.add_argument('-topo', '--topology', dest='topo', default=None,
                    help='MD topology-file name in the MD-output-directory, '\
                    'if trajectory-file does not specify topology. DEFAULT: None.',
                    type=str)
    
parser.add_argument('-fes', '--fes', dest='fes', default=None,
                    help='FES-file name in the MD-output-directory. '\
                    'DEFAULT: None',
                    type=str)

parser.add_argument('-cv', '--colvar', dest='colvar', default='COLVAR', nargs='+',
                    help='COLVAR-file in the MD-output-directory. DEFAULT: "COLVAR".',
                    type=str)

parser.add_argument('-col, --column', dest='column', default=None,
                    help='Columns for CV1;CV2 (colvar) or CV1;CV2;Energy (fes). Expects 2 integers '\
                        '(custom colvar columns), 3 integers (custom fes columns) or 5 integers (custom colvar and fes) '\
                            'delimited by commas. DEFAULT: "1,2 and 1,2,3"')

# TUNING
parser.add_argument('-thresh', '--thresh', dest='thresh', default=None,
                    help='Specifies threshold for assigning. Input value has to correspond with values in FES-file. '\
                    'DEFAULT: Lowest 1/12 of the energy span.',
                    type=float)
    
parser.add_argument('-pbc', '--pbc', dest='pbc', default='False', choices=('True','False'),
                    help='Specifies whether the FES is periodic. Expects True/False. DEFAULT: False.',
                    type=str)
    
parser.add_argument('-mindist', '--mindist', dest='mindist', default=None,
                    help='Smallest allowed distance at which areas are considered separate minima (in CV units). '\
                    'DEFAULT: 2%% of FES diagonal.',
                    type=float)

# OPTIONAL
parser.add_argument('-stride', '--stride', dest='stride', default=1,
                    help='Reads only every n-th frame of trajectory. DEFAULT: 1.',
                    type=int)

parser.add_argument('-png', '--png', dest='fes_png', default='True', choices=('True','False','only'),
                    help='Specifies whether a PNG-visualization of the FES should be created (True/False) '\
                    'or if only a PNG is desired (only). DEFAULT: True.',
                    type=str)
    
parser.add_argument('-dim', '--dim', dest='dims', default='500,500',
                    help='x-dimension,y-dimension of generated FES '\
                    'if no FES-file provided. DEFAULT: 500,500',
                    type=str)

parser.add_argument('-md', '--md', dest='md_dir', default=os.getcwd(),
                    help='Working path. DEFAULT: Current directory path.',
                    type=str)

args = parser.parse_args()


def init_polygon(_all,_x,_y,_mp,_gp,_params, _bins,_mindist, barargs):
    tqdm.tqdm.set_lock(barargs)
    global mp_sorted_coords, grouped_pts
    global all_points, pol_bins, a,b 
    global parameters, mindist
    all_points,a,b, mp_sorted_coords, grouped_pts, parameters, pol_bins, mindist = _all,_x,_y, _mp, _gp, _params, _bins, _mindist


def init_custom_writer(_sorted_indx,_fmt,barargs):
    tqdm.tqdm.set_lock(barargs)
    global sorted_indx, format
    sorted_indx, format = _sorted_indx, _fmt


def det_min_frames(j):
    polygon = shapely.Polygon(grouped_pts[j])
    convex_hull = polygon.convex_hull
    try:
        if abs(1-(polygon.area/convex_hull.area)) > 0.4:
            polygon = convex_hull
            stdout(f'polygon {j} did not initialize properly, using convex-hull')
    except ZeroDivisionError:
        pass

    pol_fill = [fes_bin for fes_bin in pol_bins if polygon.distance(shapely.Point(fes_bin)) <= mindist]
    pol_fill_keys = pos_polygon(parameters, pol_fill)
    indxes = [] 
    tol = np.sqrt(parameters[3]**2+parameters[2]**2)
    with tqdm.tqdm(total=len(pol_fill_keys), desc=f'min {j}', position=tqdm.tqdm._get_free_pos()+j, leave=False) as pbar:
        for key in pol_fill_keys:
            try:
                for point in all_points[key]:
                    if polygon.distance(shapely.Point([a[point],b[point]])) <= tol:
                        indxes.append(point)
            except KeyError:
                pass
            pbar.update(1)
        mp_sorted_coords[j] = indxes
    return [np.abs((polygon.exterior.xy[0]-parameters[4])/((parameters[6]-parameters[4])/parameters[0])),
            parameters[1]-np.abs((polygon.exterior.xy[1]-parameters[5])/((parameters[7]-parameters[5])/parameters[1]))]


def have_common_elem(l1, l2):
    for elem in l2:
        if countOf(l1, elem) > 0:
            return True
    return False


def fes_gen_histo(a,b):
    dimX, dimY = int(args.dims.split(',')[0]), int(args.dims.split(',')[1])

    histo, xedges, yedges = np.histogram2d(a, b, bins=(dimY,dimX), range=((min(a),max(a)),(min(b),max(b))), density=True)
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
    ener2d = np.flipud(-np.log(histo.T))
    return (dimX, dimY, tolX, tolY, min_a, min_b, max_a, max_b, fullX, fullY), ener2d, coords


def fes_gen_fes(pos_cvs_fes, pos_ener):
    a_fes, b_fes, ener = np.loadtxt(args.fes, unpack=True, usecols=(*pos_cvs_fes, pos_ener), dtype=float)
    dimX, dimY, ct2 = 0, 1, 0
    count1 = count(0)
    for i in range(len(ener)):
        if not np.isfinite(ener[i]):
            raise Exception('Non-finite value (NaN or inf) in FES-file')
        
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
    for i in tqdm.tqdm(range(len(x)), desc='filling hashmap', leave=False):
        pos = int((x[i]-parameters[4])/bin_cutoffX) + int((y[i]-parameters[5])/bin_cutoffY) * new_grid_dimX
        hash_tab[pos].append(i)
    return hash_tab
    
    
def pos_polygon(parameters, pts):
    new_grid_dimX = int(parameters[8]/(4*np.sqrt(parameters[2]**2+parameters[3]**2)))
    bin_cutoffX = parameters[8]/new_grid_dimX
    bin_cutoffY = parameters[9]/new_grid_dimX
    pos_pol_pts = set()
    for pt in pts:
        pos = int((pt[0]-parameters[4])/bin_cutoffX) + int((pt[1]-parameters[5])/bin_cutoffY) * new_grid_dimX
        pos_pol_pts.add(pos)
        pos_pol_pts.add(pos-1)
        pos_pol_pts.add(pos+1)
        pos_pol_pts.add(pos+new_grid_dimX)
        pos_pol_pts.add(pos-new_grid_dimX)
        pos_pol_pts.add(pos+new_grid_dimX+1)
        pos_pol_pts.add(pos-new_grid_dimX-1)
        pos_pol_pts.add(pos+new_grid_dimX-1)
        pos_pol_pts.add(pos-new_grid_dimX+1)
    return pos_pol_pts


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
    

def printout(i):
    if args.stride > 1:
        sorted_indx[i] = [args.stride*q for q in sorted_indx[i]]
    try:
        ag.write(f'minima/min_{i}.{args.traj[0].split(".")[-1]}', frames=u.trajectory[sorted_indx[i]])
    except (TypeError, ValueError):
        #print(f'MDAnalysis does not support writing in {args.traj.split(".")[-1]}-format, writing in pdb-format instead')
        ag.write(f'minima/min_{i}.xyz', frames=u.trajectory[sorted_indx[i]])

      
def printout_custom(i):
    if args.stride > 1:
        sorted_indx[i] = [args.stride*q for q in sorted_indx[i]]
    end = 'cfg' if format == 'cfg' else 'pdb'
    linecount, printcount = 0, 0
    with open(f'minima/min_{i}.'+end, 'w') as minfile:
        with tqdm.tqdm(total=len(sorted_indx[i]), desc=f'writing min {i}', 
                position=tqdm.tqdm._get_free_pos()+i, leave=False) as pbar:
            for element in args.traj:
                with open(element, 'r') as ftraj:
                    for line in ftraj:
                        try:
                            if sorted_indx[i][printcount] == linecount:
                                minfile.write(line)
                                if line.startswith('END'):
                                    printcount += 1
                                    pbar.update(1)
                        except IndexError:
                            break
                        if line.startswith('END'):
                            linecount += 1


def printout_prework(end):
    lookstr = 'BEGIN_CFG' if end == 'cfg' else 'END'
    totalframes = 0
    try:
        for filename in args.traj:
            output = run(['grep','-c',f'^{lookstr}',filename], shell=False, capture_output=True, text=True)
            totalframes += int(output.stdout)
    except FileNotFoundError:
        for filename in args.traj:
            with open(filename,'r') as f:
                for line in f:
                    if line.startswith(lookstr):
                        totalframes += 1
    return totalframes


def stdout(string, center=False, end='\n', start=''):
    try:
        terminal_size = os.get_terminal_size()[0]
    except OSError:
        terminal_size = len(string)
    if not center:
        print(start+string+(terminal_size-len(string))*' ', end=end, flush=True)
    else:
        tempstr = ' '*int((terminal_size-len(string+start))/2)
        print(start+tempstr+string+tempstr, flush=True)


if __name__ == '__main__':
    start = perf_counter()
    title = '.: Free Energy Surface Trajectory Analysis - FESTA :.'
    termin = '.: terminated successfully :.'
    version = '.: histogram-capable version :.'

    stdout(title, center=True, start='\n')
    stdout(version, center=True)
    stdout(f'working directory: {args.md_dir}', start='\n')
    os.chdir(args.md_dir)

    pos_cvs_fes = (0,1)
    pos_ener = 2
    pos_cvs_colv = (0,1)
    if args.column:
        all_custom_pos = [int(pos)-1 for pos in args.column.split(',')]
        if len(all_custom_pos) == 2:
            pos_cvs_colv = all_custom_pos
        elif len(all_custom_pos) == 3:
            pos_cvs_fes = all_custom_pos[:2]
            pos_ener = all_custom_pos[2]
        else:
            pos_cvs_fes = all_custom_pos[2:4]
            pos_ener = all_custom_pos[4]
            pos_cvs_colv = all_custom_pos[:2]

    a, b = [], []
    for element in args.colvar:
        a_tmp, b_tmp = np.loadtxt(element, unpack=True, usecols=pos_cvs_colv, dtype=float, delimiter=';')
        a.append(a_tmp)
        b.append(b_tmp)
    
    a = np.concatenate(a)
    b = np.concatenate(b)
    
    if not args.fes == None:
        parameters, ener2d, coords = fes_gen_fes(pos_cvs_fes, pos_ener)
    else:
        parameters, ener2d, coords = fes_gen_histo(a,b)

    if args.stride > 1:
        a, b = a[0::args.stride], b[0::args.stride]
    
    if args.thresh == None:
        if not args.fes == None:
            args.thresh = np.max(ener2d) - abs(np.max(ener2d)-np.min(ener2d))*(1-1/12)
            stdout('automatically determined', end=' ') 
        else:
            raise Exception('Cannot use automatic threshold detection with histogram mode')
    stdout(f'threshold value: {round(args.thresh,3)} a.U.')
    
    outline, outl_vis, edge, bins = [], [], [], []
    
    tot_bin = 0
    pol_bins = []
    for i in range(parameters[0]):
        for j in range(parameters[1]):
            tot_bin += 1
            try:
                if ener2d[i,j] < args.thresh:
                    pol_bins.append([coords[i,j,0],coords[i,j,1]])
                    if (coords[i,j,0] == parameters[4] or coords[i,j,0] == parameters[6] or coords[i,j,1] == parameters[5]
                                              or coords[i,j,1] == parameters[7] or ener2d[i-1,j]>args.thresh or ener2d[i+1,j]>args.thresh 
                                              or ener2d[i,j-1]>args.thresh or ener2d[i,j+1]>args.thresh or ener2d[i+1,j+1]>args.thresh 
                                              or ener2d[i-1,j+1]>args.thresh or ener2d[i+1,j-1]>args.thresh or ener2d[i-1,j-1]>args.thresh):
                        if coords[i,j,0] == parameters[4] or coords[i,j,0] == parameters[6] or coords[i,j,1] == parameters[5] or coords[i,j,1] == parameters[7]:
                            edge.append(coords[i,j,:])
                        outline.append(coords[i,j,:])
                        bins.append(tot_bin)
            except IndexError:
                pass
    
    if args.mindist == None:
        args.mindist = np.sqrt((parameters[6]-parameters[4])**2+(parameters[7]-parameters[5])**2)*0.02
    elif args.mindist < 2*np.sqrt(parameters[2]**2+parameters[3]**2):
        raise Exception('Minimal separation distance must be larger than diagonal of a single bin.')
    
    print('reading trajectory in ... ' , end='', flush=True)
    cp2k_pdb = False
    try:
        if args.topo == None:
            if args.traj[0].split('.')[-1] == 'lammpstrj':
                u = Universe(args.traj, topology_format='LAMMPSDUMP')
            else:
                u = Universe(args.traj)
        else:
            u = Universe(args.topo, args.traj, atom_style='atomic')
        ag = u.select_atoms('all')
        format = 'MDAnalysis'
        if not int((len(u.trajectory)-1)/args.stride+1) == len(a):
            raise Exception(f'COLVAR-file and trajectory-file must have similar step length, here: {len(a)} vs {int((len(u.trajectory)-1)/args.stride+1)}')
    except (IndexError, ValueError):
        if args.traj[0].endswith('.pdb'):
            format = 'cp2k_pdb'
        elif args.traj[0].endswith('.cfg'):
            format = 'cfg'
        else:
            raise Exception('MDAnalysis does not support this topology- or trajectory-file')
        frame_count = printout_prework(format)
        if not int((frame_count-1)/args.stride+1) == len(a):
            raise Exception(f'COLVAR-file and trajectory-file must have similar step length, here: {len(a)} vs {int((frame_count-1)/args.stride+1)}')
    except FileNotFoundError:
        raise
    print('done')
    
    print('executing CCL step ... ', end='', flush=True)
    start0 = perf_counter()
    hash_list, new_dimX = hash_fes(parameters, outline, bins, args.mindist)

    grouped_points = ex3(hash_list, new_dimX, args.mindist)
    grouped_points = [groups for groups in grouped_points if len(groups)>3]
    print('done')
    stdout(f'time needed for CCL step: {round(perf_counter() - start0,3)} s')
    
    start1 = perf_counter()
    periodicity = False
    if edge and args.pbc == 'True':
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
    if periodicity:
        print('distinctive areas identified')
    else:
        print('minimum identified') if len(grouped_points) == 1 else print('minima identified')
    
    tot_min_frames = 0

    all_points = hash_colv(parameters, a, b)
    
    usable_cpu = os.cpu_count()-1 if len(grouped_points)>os.cpu_count()-1 else len(grouped_points)

    exteriors_x, exteriors_y = [], []
    mp_sorted_coords = mp.Manager().list([[] for _ in range(len(grouped_points))])
    with mp.Pool(processes=usable_cpu, initializer=init_polygon, 
                 initargs=(all_points, a, b, mp_sorted_coords, grouped_points,
                          parameters, pol_bins, args.mindist,mp.RLock())) as pool:
        for exterior in pool.map(det_min_frames, range(len(grouped_points))):
            exteriors_x.append(exterior[0])
            exteriors_y.append(exterior[1])
    sorted_indx = list(mp_sorted_coords)
    for lists in sorted_indx:
        lists.sort()
        tot_min_frames += len(lists)

    stdout(f'processed {len(a)} frames')
    stdout(f'found {tot_min_frames} minima frames')
    if tot_min_frames/len(a) > 0.9:
        stdout(f'WARNING: {round(tot_min_frames/len(a)*100)}% of frames part of minima, check if this really is what you want')
    stdout(f'time needed for minima frames identification step: {round(perf_counter() - start1,3)} s')

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
        for i in range(len(exteriors_x)):
            plt.plot(exteriors_x[i], exteriors_y[i], '.', color='white', ms=2)
        cb = plt.colorbar(label='free energy [a.U.]', format="{x:.0f}")
        tick_locator = ticker.MaxNLocator(nbins=8)
        cb.locator = tick_locator
        cb.update_ticks()    
        plt.savefig('minima/fes_visual.png',bbox_inches='tight')
        if args.fes_png == 'only':
            stdout(termin, center=True, start='\n')
            exit()
    
    desc = []
    if periodicity == True:
        sorted_coords_period, tot_pbc  = [], []
        for elem in pbc:
            desc.append(' + '.join((f'CV1: {round(np.mean(grouped_points[j], axis=0)[0],4)} '\
            f'CV2: {round(np.mean(grouped_points[j], axis=0)[1],4)}') for j in elem))
            help_list = []
            for i in elem:
                tot_pbc.append(i)
                help_list += sorted_indx[i]
            if cp2k_pdb == True:
                help_list.sort()
            sorted_coords_period.append(help_list)
        for i,elem in enumerate(sorted_indx):
            if not i in tot_pbc:
                desc.append(f'CV1: {round(np.mean(grouped_points[i], axis=0)[0],4)} '\
                f'CV2: {round(np.mean(grouped_points[i], axis=0)[1],4)}')
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
                overviewfile.writelines(f'min_{i} : CV1: {round(np.mean(grouped_points[i], axis=0)[0],4)} '\
                f'CV2: {round(np.mean(grouped_points[i], axis=0)[1],4)}\n')
    if format == 'MDAnalysis':
        for i in tqdm.tqdm(range(len(sorted_indx)), desc='printing to file', leave=False):
            printout(i)
    else:
        with mp.Pool(processes = usable_cpu, initializer=init_custom_writer, initargs=(sorted_indx,format,mp.RLock())) as pool:
            pool.map(printout_custom, range(len(sorted_indx)))

    stdout(f'time needed for postprocessing step: {round(perf_counter() - start3,3)} s')
    stdout(termin, center=True, start='\n')
