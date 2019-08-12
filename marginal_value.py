# Plexiglass box generating marginal value maps
# Concept design by Ben Bryant
# code by Ginger Kowal
# 5.15.15

import os
import sys
if 'C:/Python27/Lib/site-packages' not in sys.path:
    sys.path.append('C:/Python27/Lib/site-packages')
sys.path.append('C:/Users/Ginger/Documents/Python/csopt_Ginger_working/csopt')
from datetime import datetime as dtdatetime
import itertools
import re
import objgraph
import logging
import importlib

import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit
import pandas
import matplotlib.pyplot as plt
from osgeo import gdal
import statsmodels.api as sm
import cvxpy as cvx
import cvxopt

import pygeoprocessing.geoprocessing
import IntegerSolver as isolve

import fitters as csopt_fit
import reporting as csopt_report

from memory_profiler import profile

gdal.UseExceptions()

global _debug
global _run_InVEST

_run_InVEST = False
_debug = False


class InVEST_model:

    """Class holding important info about an InVEST model: the args to
    run it and the output file from it that we will use to measure marginal
    value.
    eval: should the model be re-run if evaluate_solution is True? i.e., do we
    expect to get different marginal values for the model when the
    implementation solution is implemented? i.e., is the model 'spatially
    interdependent'?"""

    def __init__(self, name, args, module, output, eval):
        self.name = name
        self.args = args
        self.output = output
        if module is not None:
            self.module = importlib.import_module(module)
        self.eval = eval

    def execute(self):
        self.module.execute(self.args)


class Objective:

    """Class describing an optimization objective.  May be linked to an InVEST
    model by its name.  An objective should be maximized if the output returns
    are judged to be beneficial (e.g., maximize crop yields).  An objective
    should be minimized if output returns are judged to be detrimental (e.g.,
    minimize sediment export)."""

    def __init__(self, name, weight, lower_level_target, upper_level_target,
                 target_type, maximize):
        self.name = name
        self.weight = weight  # positive: maximize. negative: minimize
        self.l_target = lower_level_target
        self.u_target = upper_level_target
        self.target_type = target_type
        self.maximize = maximize


def get_objective_weight(objective_list, model_name):
    """Return the objective weight corresponding to a model name."""

    found = False
    for objective in objective_list:
        if objective.name == model_name:
            if found:
                raise Exception("Model name found twice in objective list")
            else:
                weight = objective.weight
                found = True
    if not found:
        raise Exception("Model name not found in objective list")
    return weight


def raster_to_np_array(raster):
    """Convert a geotiff raster to a numpy array and return the array."""

    ds = gdal.Open(raster)
    band = ds.GetRasterBand(1)

    array = np.array(band.ReadAsArray())
    nodata = band.GetNoDataValue()
    if array.dtype == np.int8:
        # gdal converts signed 8 bit to unsigned
        er = "Exception: raster is of 8 bit type"
        raise Exception(er)
        # array = array.astype(np.int32, copy = False)
    # for the purposes of this script, -9999 is always NoData
    array[array == nodata] = -9999
    ds = None

    return array


def array_to_raster(array, template_raster, new_name, datatype=None):
    """Write a numpy array to raster as a GeoTIFF of filename new_name, with
    spatial extent and projection of the template raster."""

    inDs = gdal.Open(template_raster)
    if inDs is None:
        er = "Could not open template raster dataset"
        raise Exception(er)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    if datatype is not None:
        data_type = datatype
    else:
        data_type = (inDs.GetRasterBand(1)).DataType

    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(new_name, cols, rows, 1, data_type)
    if outDs is None:
        er = "Could not create new raster object"
        raise Exception(er)
    out_band = outDs.GetRasterBand(1)
    if sparse.issparse(array):
        array = array.toarray()
    out_band.WriteArray(array)
    del array

    out_band.FlushCache()
    out_band.SetNoDataValue(-9999)

    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())

    inDs = None
    outDs = None


def merge_rasters(rasters_to_merge, save_as):
    """Mosaic positive values from several rasters of the same shape together.
    If positive regions overlap, those later in the list will cover those
    earlier so that the last raster in the list will not be covered.  Saves the
    merged raster as save_as and returns nothing."""

    def merge_op(*rasters):
        raster_list = list(rasters)
        # assume rasters share size
        result_raster = np.full((raster_list[0].shape), -9999)
        # assume nodata (should not be copied) is 0
        for raster in raster_list:
            np.copyto(result_raster, raster, where=raster > 0)
        return result_raster
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
            rasters_to_merge[0])
    pygeoprocessing.geoprocessing.vectorize_datasets(
            rasters_to_merge, merge_op, save_as,
            gdal.GDT_Int32, -9999, out_pixel_size, "union",
            dataset_to_align_index=0, vectorize_op=False)


def set_nodata_areas(pdict, value_raster, mask, out_uri):
    """Set the areas of a raster falling outside a mask to -9999 so that they
    are correctly represented as NoData in ArcGIS."""

    def copy_data_areas(value_raster, mask):
        result_raster = np.full((value_raster.shape), -9999)
        np.copyto(result_raster, value_raster, where=mask != 0)
        return result_raster

    input_list = [value_raster, mask]
    inDs = gdal.Open(value_raster)
    data_type = (inDs.GetRasterBand(1)).DataType
    del inDs
    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        value_raster)

    pygeoprocessing.geoprocessing.vectorize_datasets(
        input_list, copy_data_areas, out_uri, data_type,
        -9999, out_pixel_size, "dataset",
        dataset_to_bound_index=0, vectorize_op=False)


def save_sparse_csr(filename, array):
    """Save a sparse csr matrix."""

    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    """Load a saved sparse matrix in csr format.  Stolen from above source."""

    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'],
                              loader['indptr']), shape=loader['shape'])


def get_service_scores(raster, mask=None):
    """Retrieve values from a results raster.  If a mask is supplied, only the
    values falling under the mask are returned."""

    value_array = raster_to_np_array(raster)
    value_array[value_array == -9999] = 0

    if mask is not None:
        ret_arr = value_array
        mask_arr = mask.toarray()
        ret_arr[mask_arr != 1] = 0
    else:
        ret_arr = value_array

    sparse_array = sparse.csr_matrix(ret_arr)

    return sparse_array


def calc_marginal_scores(path1, path2):
    """Calculate marginal scores as the per-pixel difference between two arrays.
    Array 2 (pointed to by path2) is subtracted from array 1 (pointed to by
    path1)."""

    scores1 = get_service_scores(path1)
    scores2 = get_service_scores(path2)

    marginal_arr = scores1 - scores2

    return marginal_arr


def calc_cost_limit(pdict, cost_list, objective_list, rau):
    """Calculate the range of costs that should be tested in order to generate
    a continuous response curve within one RAU."""

    max_cost = max(cost_list)
    rau_dir = os.path.join(pdict[u'intermediate'], 'rau_' +
                           objective_list[0].name)
    file = os.path.join(rau_dir, 'rau' + str(rau) + '.npy')
    if not os.path.isfile(file):
        raise Exception("file %s not found" % file)
    obj_data = np.load(file)
    # this ignores any pixels where intervention is not defined
    upper_limit = [obj_data.shape[0] * max_cost]
    return upper_limit


def get_mask_list_reg(raster_shape, reg_dimensions):
    """Generate a list of raster masks by regularly sampling the landscape.
    Takes the shape of the raster to be sampled and the dimensions of the
    "building block grid" as arguments.  Returns a list of masks consisting of
    regularly sampled groups of cells; together these masks cover the raster
    entirely."""

    # generate all possible "building blocks" from regular dimensions
    # e.g. for regular dimensions (2, 2), possible building blocks are
    # (0, 0), (0, 1), (1, 0), (1, 1)

    block_list = []
    combinations = [list(xrange(reg_dimensions[0])), list(xrange(
                    reg_dimensions[1]))]
    for element in itertools.product(*combinations):
        block_list.append(element)

    # for each possible building block pattern, piece it together to cover the
    # raster
    mask_list = []
    # for block in block_list:
        # mask = ummmm ?
        # mask_list.append(mask)
    # TODO finish this
    # each element of mask_list (each mask) should be of the same shape as
    # the raster, and mask values of 1 indicate pixels to apply interventions
    return mask_list


def get_mask_list_perc(raster_sort_order, percentile_list, unique=None):
    """Generate a list of raster masks where each mask identifies a group of
    pixels by position that we expect to have minimum spatial interaction.
    Pixels are sampled in percentiles by position from the raster sort order so
    that the raster sort order defines groups of pixels that are expected to
    have minimum spatial interaction; the percentile list defines how many
    groups of pixels should be selected and how large the groups should be
    relative to each other. The percentile list may be supplied as a list of
    integers less than or equal to 100, or as a list of floats less than or
    equal to 1."""

    mask_list = []
    past = []

    integer_based = [t for t in percentile_list if t > 1]
    if integer_based:
        if np.max(percentile_list) > 100:
            er = "Error: percentile list contains elements greater than 100"
            raise Exception(er)
        print "Array is integer based, converting to float"
        percentile_list = np.multiply(percentile_list, 0.01)

    perc_list_sorted = np.sort(percentile_list)

    for percentile in perc_list_sorted:
        full_subset = raster_sort_order[:(int(percentile * len(
                                        raster_sort_order)))]
        if unique:
            mask = list(set(full_subset) - set(past))
        else:
            mask = full_subset
        mask_list.append(mask)
        past = full_subset

    # TODO cumulative or unique?
    # Ben's original design document showed this as cumulative. Then when
    # selecting the marginal value to apply to each pixel, would find the first
    # mask where that pixel appeared and take the marginal value according to
    # that mask and assign it to the pixel. Why not make these masks distinct?
    # so that each pixel appears in exactly one mask?

    return mask_list


def get_raster_sort_order(unsorted_array):
    """Generate raster pixel sort order describing which pixels should receive
    interventions first.  Pixels are sampled from the landscape in percentiles
    following this order.  The elements of raster sort order give the position
    of elements in the flattened raster."""

    flattened_raster = np.ravel(unsorted_array)
    position_array = np.arange(np.ravel(unsorted_array).shape[0])

    # remove position indices that refer to NoData pixels
    position_array = position_array[flattened_raster != -9999]

    # for now, we simply shuffle pixel position randomly
    raster_sort_order = position_array
    np.random.shuffle(raster_sort_order)

    return raster_sort_order


def extract_by_mask(values, mask, index=None):
    """Extract values from a 2-d array falling under a mask.  If an index is
    supplied (for example to identify an RAU), the values identified in the
    mask by that index are extracted."""

    if sparse.issparse(values):
        value_arr = values.toarray()
    else:
        value_arr = values
    if sparse.issparse(mask):
        mask_arr = mask.toarray()
    else:
        mask_arr = mask

    if index is not None:
        match_val = index
    else:
        match_val = 1

    extracted = np.extract(mask_arr == match_val, value_arr)

    return extracted


def get_mask_list_entire(pdict, index=None, rau_raster=None):
    """Generate one mask that covers the raster entirely.  This mask can be used
    for whole-landscape-at-a-time, non-spatially-interdependent service models.
    If an index is supplied, this generates a mask that identifies pixels of a
    raster with the value of that index."""

    if index is not None and rau_raster is not None:
        arr = raster_to_np_array(rau_raster)
    else:
        arr = raster_to_np_array(pdict[u'lulc'])
    mask = np.zeros(arr.shape)

    if index is None:
        mask[arr != -9999] = 1
    else:
        mask[arr == index] = 1

    sp_mask = sparse.csr_matrix(mask)

    return [sp_mask]


def delete_geotiff(raster_name):
    """Delete a GeoTIFF including all associated files, identified as files that
    begin with the filename raster_name.tif.  Can handle raster_name argument
    ending with '.tif', or without the .tif extension.
    Returns nothing."""

    folder = os.path.dirname(raster_name)
    file_name = os.path.basename(raster_name)

    if re.search(r".tif$", file_name):
        tif_name = file_name
    else:
        tif_name = file_name + '.tif'

    files = [f for f in os.listdir(folder) if os.path.isfile(
                                                      os.path.join(folder, f))]
    pattern = r"^" + re.escape(tif_name)
    tif_files = [f for f in files if re.search(pattern, f)]
    for file in tif_files:
        os.remove(os.path.join(folder, file))


def get_allowable_lulc(lulc_arr, intervention, biophys):
    """Get list of lulc types for which an intervention is defined in the
    biophysical table.  These are the lulc types for which the intervention can
    be implemented and its marginal value calculated."""

    allowed_lulc = []

    biophys_lu = biophys.set_index('lucode')
    biophys_desc = biophys.set_index('description')

    lulc_categories = list(np.unique(lulc_arr))
    if -9999 in lulc_categories:
        lulc_categories.remove(-9999)
    if 0 in lulc_categories:
        lulc_categories.remove(0)
    for lulc in lulc_categories:
        # look up new category according to intervention and lulc type
        desc = biophys_lu.loc[lulc]['description']
        new_desc = desc + "->" + intervention
        if new_desc in biophys_desc.index:
            allowed_lulc.append(lulc)

    return allowed_lulc


def identify_undefined_pixels(pdict, intervention_list):
    """Identify pixels of land use types for which the given intervention is
    undefined (i.e., the biophysical table does not contain parameters for the
    lulc/intervention combination)."""

    rau_raster = get_rau_raster(pdict)
    rau_list = pygeoprocessing.geoprocessing.unique_raster_values_uri(
        rau_raster)
    orig_lulc = raster_to_np_array(pdict[u'lulc'])
    biophys = pandas.read_csv(pdict[u'biophys'])
    for indx in xrange(len(intervention_list)):
        # identify undefined pixels
        intervention = intervention_list[indx]
        und_arr = np.zeros(orig_lulc.shape)
        allowed_lulc = get_allowable_lulc(orig_lulc, intervention, biophys)
        for lulc in list(np.unique(orig_lulc)):
            if lulc not in allowed_lulc:
                und_arr[orig_lulc == lulc] = 1
        for rau in rau_list:
            # extract each rau by mask
            mask = get_mask_list_entire(pdict, rau, rau_raster)[0]
            extracted = extract_by_mask(und_arr, mask)
            save_as = os.path.join(pdict[u'intermediate'],
                                   'undefined_pixels_i%d_rau%d.npy' %
                                   (indx, rau))
            np.save(save_as, extracted)


def discard_undefined_pixels(raw_model_data, undefined_pixels, weight):
    """Change the value of pixels for which a transition is undefined so that
    that intervention will never be selected by the optimizer for that
    pixel."""

    # This function is deprecated and replaced by code interior to the function
    # optimize in IntegerSolver.py.

    if weight < 0:          # objective should be minimized
        new_val = 99999
    elif weight > 0:    # objective should be maximized
        new_val = -99999
    else:
        new_val = 0
    model_arr = np.asarray(raw_model_data)
    undef_arr = np.asarray(undefined_pixels)
    assert model_arr.shape == undef_arr.shape, """Model data and undefined pixel
        data must reflect same number of pixels and interventions"""
    model_arr[undef_arr == 1] = new_val
    del undef_arr
    model_data = np.column_stack(model_arr)
    return model_data


def create_intervention_maps(pdict, model_list, intervention_list):
    """Create landuse maps where interventions are applied across the entire
    landscape.  These are the same maps that would be created if mask type ==
    'entire'.  These maps are used to translate solution rasters to lulc
    rasters following optimization."""

    orig_lulc = raster_to_np_array(pdict[u'lulc'])
    mask = get_mask_list_entire(pdict)[0]
    biophys = pandas.read_csv(pdict[u'biophys'])
    for i_index in xrange(len(intervention_list)):
        intervention = intervention_list[i_index]

        # TODO consider we are duplicating the mask_entire if num mask = 1
        # (both in processing time and storage space). Could write a clever way
        # to retrieve the mask if num mask = 1.

        # apply intervention across landscape completely
        lulcdir = os.path.join(pdict[u'outerdir'], 'modified_lulc')
        if not os.path.exists(lulcdir):
            os.makedirs(lulcdir)
        save_as = os.path.join(lulcdir, 'lulc_entire_i%d.tif' % i_index)
        apply_intervention(pdict, biophys, pdict[u'lulc'], mask, intervention,
                           save_as, delete_existing=True)


def apply_intervention(pdict, biophys, lulc_raster, mask, intervention,
                       save_as, delete_existing=False):
    """Generate a new lulc raster with the selected intervention applied to the
    raster cells that fall under the mask, saved as save_as.
    Returns nothing."""

    lulc_arr = raster_to_np_array(lulc_raster)
    if sparse.issparse(mask):
        mask = mask.toarray()
    if os.path.isfile(save_as):
        if delete_existing:
            delete_geotiff(save_as)
        else:
            er = "Error: modified lulc raster already exists"
            raise Exception(er)
    gdal.AllRegister()
    biophys_lu = biophys.set_index('lucode')
    biophys_desc = biophys.set_index('description')
    allowed_lulc = get_allowable_lulc(lulc_arr, intervention, biophys)
    arr_copy = lulc_arr.astype(np.int32, copy=True)
    lulc_list = list(np.unique(lulc_arr[mask == 1]))
    del lulc_arr
    lulc_categories = set(lulc_list).intersection(allowed_lulc)
    # TODO add list of lulc categories that are not allowed to log file
    for lulc in lulc_categories:
        # look up new category according to intervention and lulc type
        desc = biophys_lu.loc[lulc]['description']
        new_desc = desc + "->" + intervention
        new_lulc = biophys_desc.loc[new_desc]['lucode']
        arr_copy[(arr_copy == lulc) & (mask == 1)] = new_lulc
    lulc_reclassified = os.path.join(pdict[u'intermediate'],
                                     'reclassified_lulc.tif')
    temp_mask = os.path.join(pdict[u'intermediate'], 'temporary_mask.tif')
    array_to_raster(arr_copy, lulc_raster, lulc_reclassified, datatype=5)
    array_to_raster(mask, lulc_raster, temp_mask, datatype=5)
    set_nodata_areas(pdict, lulc_reclassified, temp_mask, save_as)
    delete_geotiff(lulc_reclassified)
    delete_geotiff(temp_mask)


def run_model(pdict, model, intervention_list, mask_list):
    """Create mask, apply interventions, and run the model."""

    # default run of the model
    model.execute()

    biophys = pandas.read_csv(pdict[u'biophys'])
    outdir = os.path.join(pdict[u'outerdir'], model.name, 'modified_lulc')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for i_index in xrange(len(intervention_list)):
        intervention = intervention_list[i_index]
        for m_index in xrange(len(mask_list)):
            mask = mask_list[m_index]

            # apply intervention for cells in raster mask
            suffix = '_i%d_m%d' % (i_index, m_index)
            save_as = os.path.join(outdir, 'lulc' + suffix + '.tif')
            apply_intervention(pdict, biophys, pdict[u'lulc'], mask,
                               intervention, save_as, delete_existing=True)
            model.args['lulc_uri'] = save_as

            # run model
            model.args['results_suffix'] = suffix
            model.execute()


def align_results(pdict, model, num_intervention, num_mask):
    """Align all results rasters with original lulc raster, so that later the
    results arrays can be reclassified based on lulc values at each pixel."""

    results_list = []
    # original lulc: everything aligns to this
    results_list.append(pdict[u'lulc'])
    # default model run
    results_list.append(os.path.join(model.args[u'workspace_dir'],
                                     model.output))

    # find all intervention results
    for i_index in xrange(num_intervention):
        for m_index in xrange(num_mask):
            suffix = '_i%d_m%d' % (i_index, m_index)
            rastername = model.output[:-4] + suffix + '.tif'
            results_list.append(os.path.join(model.args[u'workspace_dir'],
                                rastername))

    intermediate_dir = os.path.join(model.args[u'workspace_dir'],
                                    'aligned_data')
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    dataset_names = [os.path.basename(file) for file in results_list]
    aligned_list = [os.path.join(intermediate_dir, name) for name in
                    dataset_names]

    out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                                                               results_list[0])
    pygeoprocessing.geoprocessing.align_dataset_list(
        results_list, aligned_list, ['nearest'] * len(aligned_list),
        out_pixel_size, 'dataset', 0, dataset_to_bound_index=0)

    return aligned_list


def calc_soln_agg_obj(pdict, model_list, objective_list, rau_list,
                      intervention_list, value_db, row=None, rsuf=None):
    """Calculate the actual aggregate objective achieved following
    implementation of the optimal solution by creating lulc maps that reflect
    the solution and running InVEST models on these maps.  The aggregate
    objective is calculated from model results per RAU and can be compared to
    the aggregate objective predicted by the upper level solver."""

    # create intervention maps to support translation of solution raster to
    # lulc
    create_intervention_maps(pdict, model_list, intervention_list)

    # translate solution rasters to modified lulc
    lulc_rasters = [pdict[u'lulc']]
    soln_rasters = []
    for rau in rau_list:
        if row is None:
            soln = os.path.join(pdict[u'sol_map_dir'], 'solution_rau%d.tif' %
                                int(rau))
        else:
            soln = os.path.join(pdict[u'sol_map_dir'],
                                'solution_rau%d_row%d.tif' % (int(rau), row))
        lulc = pdict[u'lulc']

        input_list = [soln, lulc]
        if pygeoprocessing.geoprocessing.unique_raster_values_uri(soln) != [0]:
            soln_rasters.append(soln)
        for i_index in range(len(intervention_list)):
            input_list.append(os.path.join(pdict[u'outerdir'], 'modified_lulc',
                              'lulc_entire_i%d.tif' % i_index))
        result_raster = os.path.join(pdict[u'outerdir'], 'modified_lulc',
                                     'soln_lulc_rau%d.tif' % int(rau))
        lulc_rasters.append(result_raster)

        out_pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
                                                                pdict[u'lulc'])

        def solution_to_lulc(soln, lulc, *args):
            # a list with unknown length; must be sorted
            intervention_rasters = list(args)
            # intervention 0 ---> do nothing
            intervention_rasters.insert(0, lulc)

            result_raster = np.full((soln.shape), -9999)

            for intervention in xrange(len(intervention_rasters)):
                np.copyto(result_raster, intervention_rasters[intervention],
                          where=soln == intervention)
            return result_raster
        pygeoprocessing.geoprocessing.vectorize_datasets(
            input_list, solution_to_lulc, result_raster,
            gdal.GDT_Int32, -9999, out_pixel_size, "union",
            dataset_to_align_index=0, vectorize_op=False)

    if row is None:
        if rsuf is None:
            merged_lulc = os.path.join(pdict[u'sol_map_dir'],
                                       'merged_soln_lulc.tif')
            merged_soln = os.path.join(pdict[u'sol_map_dir'],
                                       'merged_soln.tif')
        else:
            merged_lulc = os.path.join(pdict[u'sol_map_dir'],
                                       'merged_soln_lulc_%s.tif' % rsuf)
            merged_soln = os.path.join(pdict[u'sol_map_dir'],
                                       'merged_soln_%s.tif' % rsuf)
    else:
        if rsuf is None:
            merged_lulc = os.path.join(pdict[u'sol_map_dir'],
                                       'merged_soln_lulc_row%d.tif' % row)
            merged_soln = os.path.join(pdict[u'sol_map_dir'],
                                       'merged_soln_row%d.tif' % row)
        else:
            merged_lulc = os.path.join(pdict[u'sol_map_dir'],
                                       'merged_soln_lulc_row%d_%s.tif' %
                                       (row, rsuf))
            merged_soln = os.path.join(pdict[u'sol_map_dir'],
                                       'merged_soln_row%d_%s.tif' %
                                       (row, rsuf))
    merge_rasters(lulc_rasters, merged_lulc)
    if len(soln_rasters) > 0:
        merge_rasters(soln_rasters, merged_soln)

    soln_margv = {}
    for model in model_list:
        if not model.eval:
            # we don't expect that re-running the model would make a difference
            # (predicted == realized)
            # populate soln_margv directly from value_db
            soln_margv['realized_' + model.name] = []
            for idx in xrange(len(value_db)):
                soln_margv['realized_' + model.name].append(
                                                  value_db[idx][model.name][0])
            continue

        # run InVEST with solution rasters, collect actual marginal value
        model.args[u'lulc_uri'] = merged_lulc
        suffix = 'optim_soln'
        model.args['results_suffix'] = suffix
        model.execute()
        baseline = os.path.join(model.args[u'workspace_dir'], model.output)
        soln = os.path.join(model.args[u'workspace_dir'],
                            model.output[:-4] + '_' + suffix +
                            model.output[-4:])
        marginal_scores = calc_marginal_scores(soln, baseline)  # soln-baseline
        rau_raster = get_rau_raster(pdict)
        rau_mask = sparse.csr_matrix(raster_to_np_array(rau_raster))
        model_margv = []
        for rau in rau_list:
            extr_vals = extract_by_mask(marginal_scores, rau_mask,
                                        index=int(rau))
            model_margv.append(np.sum(extr_vals))
        soln_margv["realized_" + model.name] = model_margv

    modeled_objs = [model.name for model in model_list]
    for objective in objective_list:
        if objective.name not in modeled_objs:
            # get rau value some other way
            soln_margv["realized_" + objective.name] = [0] * len(rau_list)

    # calc aggregate objective
    soln_margv['realized_agg_obj'] = []
    for ridx in xrange(len(rau_list)):  # for each rau
        rau_agg_obj = 0
        for objective in objective_list:
            rau_agg_obj += (objective.weight * soln_margv["realized_" +
                            objective.name][ridx])
        soln_margv['realized_agg_obj'].append(rau_agg_obj)
    return soln_margv


def get_intervention_scores(pdict, model, baseline_raster_path,
                            result_raster_path, orig_lulc, intervention,
                            mask_entire):
    """Get marginal values from one InVEST run."""

    # calculate marginal scores
    marginal_scores = calc_marginal_scores(result_raster_path,
                                           baseline_raster_path)

    if _debug:
        # check the final array
        debugdir = os.path.join(model.args[u'workspace_dir'], 'debug')
        if not os.path.exists(debugdir):
            os.makedirs(debugdir)
        suffix = os.path.basename(result_raster_path)[11:]

        marg_arrtest = marginal_scores.toarray()
        mname = 'marginal' + suffix
        array_to_raster(marg_arrtest, result_raster_path, os.path.join(
                        debugdir, mname))
        del marg_arrtest

    # remove all nodata cells outside raster
    extracted_scores = extract_by_mask(marginal_scores, mask_entire)
    if _debug:
        save_as = os.path.join(debugdir, 'extracted' + suffix)
        values_to_raster(pdict, extracted_scores, save_as, index=None,
                         use_existing=False)

    return extracted_scores


def collect_results(pdict, model, aligned_results_list, intervention_list):
    """Collect results from InVEST runs and format for input to the
    optimizer."""

    orig_lulc_arr = raster_to_np_array(pdict[u'lulc'])
    orig_lulc_arr[orig_lulc_arr == -9999] = 0
    orig_lulc = sparse.csr_matrix(orig_lulc_arr)
    del orig_lulc_arr

    mask_entire = get_mask_list_entire(pdict)[0]

    intervention_scores_list = []

    for i_index in xrange(len(intervention_list)):
        intervention = intervention_list[i_index]
        default_raster_path = aligned_results_list[1]

        pattern = r"_i" + str(i_index)
        result_rasters = [p for p in aligned_results_list if
                          re.search(pattern, p)]
        # TODO handle multiple masks that need to be mosaicked together (will
        # be the rasters in 'result_rasters')

        # for now
        result_raster_path = result_rasters[0]

        scores = get_intervention_scores(pdict, model, default_raster_path,
                                         result_raster_path, orig_lulc,
                                         intervention, mask_entire)
        if np.nanmax(scores) == 0 and np.nanmin(scores) == 0:
            print "Warning: intervention scores are all zero"
        intervention_scores_list.append(scores)
    return intervention_scores_list


def get_ll_data_from_rasters(pdict, model, intervention_list, mask_type):
    """Collect data from results rasters that were created independently of an
    InVEST model.  The rasters, located in the folder specified by
    model.args[u'workspace_dir'], should have the name model.output followed by
    a suffix of the form 'i_X_m_0' where X is in the set of indices of the
    intervention_list."""

    print "Generating marginal values from rasters....."
    intermediate_dir = os.path.join(pdict[u'outerdir'], 'intermediate')
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    pdict[u'intermediate'] = intermediate_dir

    raw_model_data = get_ll_data(pdict, model, intervention_list, mask_type)
    model_arr = np.asarray(raw_model_data)
    model_data = np.column_stack(model_arr)

    model_data_file = os.path.join(pdict[u'intermediate'], model.name +
                                   '_ll_data.npy')
    np.save(model_data_file, model_data)
    del model_data
    del model_arr
    del raw_model_data


def get_ll_data(pdict, model, intervention_list, mask_type):
    """Generate lower level data for input to the optimizer by running InVEST
    models with different interventions applied."""

    if mask_type == 'entire':
        num_mask = 1
    else:
        num_mask = 1  # placeholder: determine # masks here from mask method

    results_list = []
    mask_range = xrange(num_mask)
    intervention_range = xrange(len(intervention_list))
    for mindx in mask_range:
        for indx in intervention_range:
            suffix = '_i%d_m%d' % (indx, mindx)
            rastername = model.output[:-4] + suffix + '.tif'
            results_list.append(os.path.join(model.args[u'workspace_dir'],
                                rastername))
    found_results = [f for f in results_list if os.path.isfile(f)]
    if len(found_results) < len(results_list):
        # fancy mask generation goes here
        mask_list = get_mask_list_entire(pdict)
        run_model(pdict, model, intervention_list, mask_list)
        del mask_list
    aligned_results_list = align_results(pdict, model, len(intervention_list),
                                         num_mask)

    ll_data = collect_results(pdict, model, aligned_results_list,
                              intervention_list)
    return ll_data


def values_to_raster(pdict, values, save_as, index=None, use_existing=False):
    """Create a raster with values from an array corresponding to the area
    identified by a mask.  Useful for testing/visualizing intermediate results.
    This formulation assumes there is one value for each pixel in the mask."""

    if index is None:
        mask_uri = pdict[u'lulc']
    else:
        mask_uri = os.path.join(pdict[u'intermediate'], 'temp_mask.tif')
        template_raster = pdict[u'lulc']
        rau_raster = os.path.join(pdict[u'intermediate'], 'rau.tif')
        mask_arr = get_mask_list_entire(pdict, index, rau_raster)[0]
        array_to_raster(mask_arr, template_raster, mask_uri, datatype=5)
        del mask_arr

    filename = os.path.basename(save_as)[:-4]
    data_file = os.path.join(pdict[u'intermediate'], filename + '.npz')
    if os.path.isfile(data_file) and use_existing:
        val_raster = load_sparse_csr(data_file)
    else:
        if index is None:
            mask = get_mask_list_entire(pdict)[0]
        else:
            rau_raster = os.path.join(pdict[u'intermediate'], 'rau.tif')
            mask = get_mask_list_entire(pdict, index, rau_raster)[0]

        n_pixels = mask.nnz
        assert values.shape[0] == n_pixels, """There should be one value for
                                               each pixel"""

        val_raster = sparse.lil_matrix(mask.shape, dtype=values.dtype)
        pidx = zip(*mask.nonzero())
        for i in xrange(n_pixels):
            val_raster[pidx[i]] = values[i]  # TODO this is amazingly slow!!!!
        val_raster = val_raster.tocsr()
        save_sparse_csr(data_file, val_raster)
        del pidx
        del mask

    template_raster = pdict[u'lulc']
    temp_name = filename + '_temp.tif'
    array_to_raster(val_raster, template_raster, temp_name, datatype=6)
    set_nodata_areas(pdict, temp_name, mask_uri, save_as)
    delete_geotiff(mask_uri)


def extract_aoi(values, pdict, aoi, rau_i=None):
    """Extract values identified by a raster as from a small area of interest.
    If rau_i is supplied, only values in the aoi raster identified by that
    number are extracted."""

    # TODO this is very inefficient - we build the *entire* value raster first
    # then extract the relevant values from it. Would be better to build the
    # entire raster once, extract values from each rau_i sequentially

    mask_entire = get_mask_list_entire(pdict)[0]
    n_pixels = mask_entire.nnz
    assert values.shape[0] == n_pixels, """There should be one value for each
                                           pixel"""
    val_raster = sparse.lil_matrix(mask_entire.shape, dtype=values.dtype)
    pidx = zip(*mask_entire.nonzero())
    for i in xrange(n_pixels):
        val_raster[pidx[i]] = values[i]  # TODO this is amazingly slow!!!!
    del mask_entire

    aoi_mask = sparse.csr_matrix(raster_to_np_array(aoi))
    extracted_vals = extract_by_mask(val_raster, aoi_mask, rau_i)
    return extracted_vals


def get_rau_list_from_files(pdict, model_list):
    """Find the RAUs for which model data has been generated and extracted."""

    rau_list_by_model = {}
    for model in model_list:
        rau_list_by_model[model.name] = []

        # Ensure all models have data for each RAU
        rau_dir = os.path.join(pdict[u'intermediate'], 'rau_' + model.name)
        if not os.path.exists(rau_dir):
            continue
        files = [f for f in os.listdir(rau_dir) if os.path.isfile(
                                                     os.path.join(rau_dir, f))]
        rau_files = [f for f in files if re.search('rau', f)]
        for file in rau_files:  # for each RAU
            try:
                rau = int(re.search('rau(.+?).npy', file).group(1))
            except AttributeError:
                raise Exception("RAU index not identified from file %s" % file)
            rau_list_by_model[model.name].append(rau)
    return rau_list_by_model


def get_solution_db(pdict, objective_list, model_list, cost_list,
                    intervention_list, mask_type, use_existing,
                    tables_list=None):
    """Run integer optimization for each RAU, get the lower level curves that
    will be subjected to upper level optimization.  If the curves have already
    been generated and saved as csv and use_existing == True, the curves are
    loaded instead of being generated."""

    loaded_from_file = False
    rau_list = get_rau_data(pdict, objective_list, model_list, cost_list,
                            intervention_list, mask_type, use_existing)
    ll_value_db = []
    if tables_list is None:
        for rau in rau_list:
            filename = os.path.join(pdict[u'output'],
                                    'solution_curve_rau%d.csv' % rau)
            if os.path.isfile(filename) and use_existing:
                df = pandas.read_table(filename, sep=',')
                value_db_RAU = df.to_dict(outtype='list')
                if 'Unnamed: 0' in value_db_RAU.keys():
                    del value_db_RAU['Unnamed: 0']
                ll_value_db.append(value_db_RAU)
                loaded_from_file = True
    if loaded_from_file:
        assert len(ll_value_db) == len(rau_list), """Must load lower level value
            db for each RAU"""
        print "loading lower-level curves from file ..."
    else:
        for rau in rau_list:
            cost_limit = calc_cost_limit(pdict, cost_list, objective_list, rau)
            test_db = get_ll_solution(pdict, rau, objective_list, cost_limit)
            upper_feasible_limit = test_db['cost'][0]
            cost_range = np.linspace(upper_feasible_limit * 0.05,
                                     upper_feasible_limit, num=10)
            value_db_RAU = get_ll_solution(pdict, rau, objective_list,
                                           cost_range, tables_list=tables_list)
            ll_value_db.append(value_db_RAU)
    return rau_list, ll_value_db


def normalize_values(folder, maximize):
    """Normalize marginal values across RAUs within an objective.  All values
    should range between 0 and 1.  It is assumed that the marginal values
    within one objective across RAUs all reside in the folder which is given as
    input."""

    savedir = os.path.join(folder, 'norm')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    filenames = []
    raw_arrays = []
    split_indices = []
    rau_files = [f for f in os.listdir(folder) if os.path.isfile(
                 os.path.join(folder, f))]
    for file in rau_files:
        # right now, assume we just want to take all files in a folder
        filenames.append(file)
        ar = np.load(os.path.join(folder, file))
        raw_arrays.append(ar)
        if len(split_indices) == 0:
            split_indices.append(ar.shape[0])
        else:
            split_indices.append(split_indices[-1] + ar.shape[0])
    joined_ar = np.concatenate(raw_arrays)
    maxval = np.max(joined_ar)
    minval = np.min(joined_ar)
    if maximize:
        result_ar = (joined_ar - minval) / (maxval - minval)
    else:
        result_ar = (maxval - joined_ar) / (maxval - minval)
    # assert np.min(result_ar) == 0, "Normalization did not work correctly"
    # assert np.max(result_ar) == 1, "Normalization did not work correctly"
    norm_arrays = np.split(result_ar, split_indices)
    del norm_arrays[-1]
    for ar_indx in xrange(len(norm_arrays)):
        assert norm_arrays[ar_indx].shape == raw_arrays[ar_indx].shape
        save_as = os.path.join(savedir, filenames[ar_indx])
        np.save(save_as, norm_arrays[ar_indx])


def get_ll_solution(pdict, rau, objective_list, cost_range, save_as=None,
                    tables_list=None, row=None, un_norm=False):
    """Get lower level solution for one RAU, under one or a range of cost
    constraints.  If save_as is supplied, a raster of the solution is saved
    with that filename.  Returns a dictionary composed of two lists, one giving
    the value of the aggregate objective and another giving the cost of the
    lower level solution. These lists correspond in their order, such that the
    first value corresponds to the first cost.  If un_norm is True, the values
    returned correspond to non-normalized (untransformed) marginal values, in
    the units of the marginal value calculations.  Otherwise, normalized values
    are assumed."""

    ll_problem = {'weights': {},
                  'targets': {},
                  'targettypes': {}
                  }
    for objective in objective_list:
        ll_problem['weights'][objective.name] = objective.weight
        if objective.l_target is not None:
            ll_problem['targets'][objective.name] = objective.l_target
        if objective.target_type is not None:
            ll_problem['targettypes'][objective.name] = objective.target_type

    value_db_RAU = {'budget': cost_range, 'agg_obj': [], 'num_IUs': [],
                    'num_IU_converted': []}
    ll_data = {'factornames': []}
    for objective in objective_list:
        ll_data['factornames'].append(objective.name)
        value_db_RAU[objective.name] = []
        if tables_list is None:
            if objective.name == 'cost' or objective.name == 'Cost':
                rau_dir = os.path.join(pdict[u'intermediate'], 'rau_' +
                                       objective.name)
            else:
                rau_dir = os.path.join(pdict[u'intermediate'], 'rau_' +
                                       objective.name, 'norm')
            file = os.path.join(rau_dir, 'rau' + str(rau) + '.npy')
            if not os.path.isfile(file):
                raise Exception("file %s not found" % file)
        else:
            match_string = objective.name + "_rau%d" % rau
            pattern = r"." + re.escape(match_string)
            files = [f for f in tables_list if re.search(pattern, f)]
            assert len(files) == 1, "One table must match search query"
            file = files[0]
        ll_data[objective.name] = np.load(file)

    if un_norm:
        ll_marg_data = ll_data.copy()
        for objective in objective_list:
            if tables_list is None:
                rau_dir = os.path.join(pdict[u'intermediate'], 'rau_' +
                                       objective.name)
                file = os.path.join(rau_dir, 'rau' + str(rau) + '.npy')
                if not os.path.isfile(file):
                    raise Exception("file %s not found" % file)
            else:
                # TODO consider normalization of supplied marg value tables
                # (this formulation makes no distinction between normalized
                # and non-normalized supplied tables)
                match_string = objective.name + "_rau%d" % rau
                pattern = r"." + re.escape(match_string)
                files = [f for f in tables_list if re.search(pattern, f)]
                assert len(files) == 1, "One table must match search query"
                file = files[0]
            ll_marg_data[objective.name] = np.load(file)
    else:
        ll_marg_data = None

    # load undefined pixels
    undefined_ar_list = []
    undefined_ar_list.append([0] * ll_data[objective_list[0].name].shape[0])
    num_interventions = ll_data[objective_list[0].name].shape[1] - 1
    for indx in xrange(num_interventions):
        intervention_ar = np.load(os.path.join(pdict[u'intermediate'],
                                  'undefined_pixels_i%d_rau%d.npy' % (indx,
                                                                      rau)))
        undefined_ar_list.append(intervention_ar)
    undefined_array = np.column_stack(undefined_ar_list)

    value_db_RAU['num_IUs'] = [ll_data[objective_list[0].name].shape[0]] * len(
        cost_range)
    for idx in xrange(len(cost_range)):
        cost_constraint = cost_range[idx]
        ll_problem['targets']['cost'] = cost_constraint
        print "entering ll optimization: RAU %d, cost constraint %f" % (int(
                                                         rau), cost_constraint)
        solution, scores = integer_optimization(ll_data, ll_problem, rau,
                                                undefined_array, ll_marg_data)
        if solution is not None:
            value_db_RAU['agg_obj'].append(scores['objective'])
            value_db_RAU['num_IU_converted'].append(np.count_nonzero(solution))
            for factor in ll_data['factornames']:
                value_db_RAU[factor].append(scores[factor])
            if save_as is not None:
                values_to_raster(pdict, np.array(solution), save_as,
                                 index=int(rau), use_existing=False)
        else:
            value_db_RAU['agg_obj'].append([])
            value_db_RAU['num_IU_converted'].append([])
            for factor in ll_data['factornames']:
                value_db_RAU[factor].append([])
        del solution
        del scores
    return value_db_RAU


def check_curve(value_db_RAU):
    """Check the points of the lower level curve for convexity and
    monotonicity.  If either of these is violated, return a warning."""

    warn_monot = False
    warn_convex = False
    convex_u = False
    convex_d = False
    slope_list = []
    for idx in xrange(len(value_db_RAU['agg_obj']) - 1):
        slope = (value_db_RAU['agg_obj'][idx + 1] -
                 value_db_RAU['agg_obj'][idx]) / (value_db_RAU['cost']
                 [idx + 1] - value_db_RAU['cost'][idx])
        if slope < 0:
            warn_convex = True
        slope_list.append(slope)
    for idx in xrange(len(slope_list) - 1):
        if slope_list[idx + 1] == slope_list[idx]:
            continue
        if slope_list[idx + 1] > slope_list[idx]:
            convex_u = True
            if convex_d:
                warn_monot = True
        else:
            convex_d = True
            if convex_u:
                warn_monot = True
    return warn_convex, warn_monot


def get_rau_data(pdict, objective_list, model_list, cost_list,
                 intervention_list, mask_type, use_existing):
    """Get data to supply lower-level solver for each RAU, for all modelled
    objectives and cost."""

    rau_raster = get_rau_raster(pdict)
    rau_list_from_raster = pygeoprocessing.geoprocessing.unique_raster_values_uri(
                                           rau_raster)
    for value in rau_list_from_raster:
        assert value - int(value) == 0, "RAU ids must be integers"
    rau_list_from_raster = [int(value) for value in rau_list_from_raster]

    rau_list_by_model = get_rau_list_from_files(pdict, model_list)
    for model in model_list:
        if rau_list_by_model is None:
            rau_to_generate = rau_list_from_raster
        else:
            rau_list = rau_list_by_model[model.name]
            rau_to_generate = set(rau_list_from_raster) - set(rau_list)
        if len(rau_to_generate) > 0:
            # load or generate model results data
            model_data_file = os.path.join(pdict[u'intermediate'], model.name +
                                           '_ll_data.npy')
            if os.path.isfile(model_data_file) and use_existing:
                model_data = np.load(model_data_file)
            else:
                raw_model_data = get_ll_data(pdict, model, intervention_list,
                                             mask_type)
                model_arr = np.asarray(raw_model_data)
                model_data = np.column_stack(model_arr)
                np.save(model_data_file, model_data)

            if _debug:
                # check rasters (Careful! this uses a ton of memory and take
                # forever)
                debugdir = os.path.join(pdict[u'intermediate'], 'debug')
                if not os.path.exists(debugdir):
                    os.makedirs(debugdir)
                for i_index in xrange(len(intervention_list)):
                    intervention = intervention_list[i_index]
                    values = model_data[:, i_index]
                    raster_name = os.path.join(debugdir, intervention +
                                               '_vals.tif')
                    values_to_raster(debugdir, values, raster_name,
                                     use_existing=False)

            # Generate per-RAU marginal value data
            extract_rau_vals(pdict, model, model_data, rau_to_generate)
            del model_data
        rau_list_by_model = get_rau_list_from_files(pdict, model_list)
    for model in model_list:
        rau_list = rau_list_by_model[model.name]
        assert rau_list == rau_list_from_raster, """Extraction of RAU vals
                                                    did not work correctly"""
    copy_dir = os.path.join(pdict[u'intermediate'], 'rau_' +
                            model_list[0].name)
    make_cost_data(pdict, cost_list, copy_dir)
    for objective in objective_list:
        if objective.name == 'cost' or objective.name == 'Cost':
            continue
        folder = os.path.join(pdict[u'intermediate'], 'rau_' + objective.name)
        normalize_values(folder, objective.maximize)
    return rau_list


def integer_optimization(data, problem, rau, undefined_array=None,
                         marg_data=None, tiebreaker_intervention=None):
    """Lower level optimization solver.  Adapted from work by Peter Hawthorne.
    The scores returned here are calculated from the data supplied to the
    solver by default; these are assumed to be normalized.  If marg_data is
    supplied, the optimal solution is calculated from normalized data but the
    returns from the solution are calculated from marg_data.
    If a tiebreaker intervention is supplied, it is forcibly chosen in case of
    implementation units where multiple interventions have equal objective
    values."""

    data['glpkopts'] = dict()
    data['glpkopts']['mip_gap'] = 0.05
    data['glpkopts']['tm_lim'] = 60000

    # optimize
    solution = isolve.optimize(data, problem, undefined_array,
                               tiebreaker_intervention)
    if solution is None:
        scores = None
    else:
        if marg_data is None:
            scores = isolve.values(solution, data, problem)
        else:
            factors = data['factornames']
            nparcels, nopts = np.array(data[factors[0]]).shape
            marg_data['nparcels'] = nparcels
            marg_data['nopts'] = nopts
            scores = isolve.values(solution, marg_data, problem)
    return solution, scores


def michaelis_menten_solver(params, max_cost, budget):
    """Upper-level solver assuming the params describe a Michaelis-Menten
    saturating function. Adapted from work by Peter Hawthorne."""

    va = cvxopt.matrix(params[:, 0])
    vb = cvxopt.matrix(params[:, 1])
    n = va.size[0]

    Gin = cvxopt.matrix(0.0, (2*n+1, n))
    hin = cvxopt.matrix(0.0, (2*n+1, 1))
    for i in range(n):
        Gin[i, i] = -1.0
        Gin[i+n, i] = 1.0
        hin[i+n] = max_cost[i]  # upper bound for each RAU
    Gin[2*n, :], hin[2*n] = 1.0, budget  # sum of x <= budget

    def Fmm(x=None, z=None):
        if x is None:
            return 0, cvxopt.matrix(float(budget)/n, (n, 1))
        if min(x) < 0.0:
            return None
        f = -1*sum(cvxopt.div(cvxopt.mul(va, x), vb+x))
        Df = -1*(cvxopt.div(va, (vb + x)) -
                 cvxopt.div(cvxopt.mul(va, x), (vb + x)**2)).T
        if z is None:
            return f, Df
        h = 2*cvxopt.div(cvxopt.mul(va, x), (vb + x)**3) - \
            cvxopt.div(2*va, (vb + x)**2)
        H = -1*cvxopt.spdiag(z[0] * h)
        return f, Df, H
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxiters'] = 10000
    x = np.squeeze(np.array(cvxopt.solvers.cp(Fmm, G=Gin, h=hin)['x']))

    def mm(x, a, b):
        return (a*x)/(b+x)

    expobj = mm(x, params[:, 0], params[:, 1])
    return x, expobj


def quad_optimization(params, budget):
    """Upper level quadratic optimization solver.  Adapted from work by Peter
    Hawthorne."""
    # TODO Peter has a more complex model formulation outlined in QPSolver.py
    # including constraints etc

    n = params.shape[0]

    x = cvx.Variable(n)
    P0 = cvx.diag(params[:, 2])
    q0 = np.column_stack(params[:, 1])
    r0 = sum(params[:, 0])

    objective = cvx.Maximize(cvx.quad_form(x, P0) + q0*x + r0)
    constraints = [sum(x) <= budget, x >= 0]  # x <= budget
    prob = cvx.Problem(objective, constraints)
    try:
        prob.solve()
        optx = x.value
    except:
        print "Warning: quadratic solver failed"
        optx = None

    # get predicted aggregate objective score for each RAU
    ypred = []
    if optx is not None:
        optx_ar = np.ravel(optx)
        for i in xrange(n):
            solx = optx_ar[i]
            ep = params[i, :]
            ypred.append(ep[0] + ep[1]*solx + ep[2]*(solx**2))

    return optx, ypred


def retrieve_implementation_solution(optx, pdict, objective_list, model_list,
                                     intervention_list, rau_list, mask_type,
                                     use_existing, save, tables_list=None,
                                     row=None):
    """Retrieve the specified implementation solution given a cost constraint
    and save the implementation solution as a raster."""

    value_db = []
    for r_indx in xrange(len(rau_list)):
        cost_range = [optx[r_indx]]
        rau = rau_list[r_indx]
        save_as = None
        if save:
            if row is None:
                save_as = os.path.join(pdict[u'sol_map_dir'],
                                       'solution_rau%d.tif' % int(rau))
            else:
                save_as = os.path.join(pdict[u'sol_map_dir'],
                                       'solution_rau%d_row%d.tif' % (int(rau),
                                       row))
        value_db_RAU = get_ll_solution(pdict, rau, objective_list, cost_range,
                                       save_as, tables_list=tables_list,
                                       un_norm=True)
        value_db.append(value_db_RAU)
    return value_db


def get_rau_raster(pdict):
    """Create a raster identifying RAUs that aligns with the lulc raster.
    rau_shp must be a shapefile containing an integer field 'rau' that
    is a unique identifier for each RAU."""

    rau_raster = os.path.join(pdict[u'intermediate'], 'rau.tif')
    template_raster = pdict[u'lulc']

    pygeoprocessing.geoprocessing.new_raster_from_base_uri(
            template_raster, rau_raster, 'GTiff', -9999, gdal.GDT_Int32,
            fill_value=-9999)
    field = 'rau'
    pygeoprocessing.geoprocessing.rasterize_layer_uri(
            rau_raster, pdict[u'rau_shp'],
            option_list=["ATTRIBUTE=%s" % field])

    return rau_raster


def extract_rau_vals(pdict, model, model_results, rau_to_generate):
    """Extract model results values from the area identified as an RAU
    (resource allocation unit)."""

    rau_raster = get_rau_raster(pdict)
    rau_dir = os.path.join(pdict[u'intermediate'], 'rau_' + model.name)
    if not os.path.exists(rau_dir):
        os.makedirs(rau_dir)
    for rau in rau_to_generate:  # for each RAU
        print "extracting values for model %s, RAU %d" % (model.name, rau)
        rau_data = []
        # for each intervention
        for col_index in xrange(model_results.shape[1]):
            vals = model_results[:, [col_index]]
            extr_vals = extract_aoi(vals, pdict, rau_raster, rau)
            if np.nanmax(extr_vals) == 0 and np.nanmin(extr_vals) == 0:
                print "Warning: extracted marginal values are all zero"
            rau_data.append(extr_vals)
            del extr_vals
        # add an intervention which is to do nothing (marginal value = 0)
        do_nothing = [0] * len(rau_data[0])
        rau_data.insert(0, do_nothing)
        model_data = np.column_stack(tuple(rau_data))
        filename = 'rau' + str(rau) + '.npy'
        filepath = os.path.join(rau_dir, filename)
        np.save(filepath, model_data)

        if _debug:
            # check rasters (Careful! this uses a ton of memory and takes 
            # forever)
            debugdir = os.path.join(pdict[u'intermediate'], 'debug')
            if not os.path.exists(debugdir):
                os.makedirs(debugdir)
            for i_index in xrange(len(model_data[0])):
                values = model_data[:, i_index]
                raster_name = os.path.join(debugdir, 'rau_' + str(rau) + 'i_' +
                                           str(i_index) + '_vals.tif')
                values_to_raster(pdict, values, raster_name, index=int(rau),
                                 use_existing=False)


def make_cost_data(pdict, cost_list, copy_dir):
    """Generate cost data for each RAU.  Uses a copy directory where real model
    results data reside to generate data for the correct shape. Relies on
    symmetrical ordering of intervention_list and cost_list."""

    cost_dir = os.path.join(pdict[u'intermediate'], 'rau_' + 'cost')
    if not os.path.exists(cost_dir):
        os.makedirs(cost_dir)
    files = [f for f in os.listdir(copy_dir) if os.path.isfile(
            os.path.join(copy_dir, f))]
    rau_files = [f for f in files if re.search('rau', f)]
    for file in rau_files:  # for each RAU
        rau = re.search('rau(.+?).npy', file).group(1)
        copy_data = np.load(os.path.join(copy_dir, file))
        assert len(cost_list) == copy_data.shape[1] - 1, """Cost must be
                defined for each intervention"""
        cost_data = copy_data
        cost_data[:, [0]] = 0  # do-nothing intervention
        for indx in xrange(len(cost_list)):
            cost_data[:, [indx + 1]] = cost_list[indx]
        filename = 'rau' + str(rau) + '.npy'
        filepath = os.path.join(cost_dir, filename)
        np.save(filepath, cost_data)


def michaelis_menten_fitter(pdict, row, rau_list, ll_value_db,
                            results_suffix=None):
    """Fit a Michaelis-Menten saturating function to data using nonlinear least
    squares."""

    x_list = []
    y_list = []
    results_list = []

    n_RAU = len(ll_value_db)   # number of RAUs
    estparams = np.zeros((n_RAU, 2))
    for i in xrange(len(ll_value_db)):
        val_dict = ll_value_db[i]
        cost = val_dict['cost']
        cost.insert(0, 0)
        obj = val_dict['agg_obj']
        obj.insert(0, 0)
        x = np.array(cost)
        y = np.array(obj)

        result = csopt_fit.michaelis_menten(y, x)
        estparams[i, :] = [result.best_values['a'], result.best_values['b']]

        x_list.append(x)
        y_list.append(y)
        results_list.append(result.best_fit)
    if results_suffix is None:
        save_as = os.path.join(pdict[u'intermediate'], 'll_curves_row%d.png'
                               % row)
    else:
        save_as = os.path.join(pdict[u'intermediate'], 'll_curves_row%d_%s.png'
                               % (row, results_suffix))
    # plot_ll_curve(rau_list, x_list, y_list, save_as,
    # results_list=results_list)
    return estparams


def quadratic_curve_fitter(pdict, ll_value_db, rau_list):
    """Fit a quadratic model to each RAU describing the aggregate objective
    score as a function of cost.  This function was adapted from Peter's
    CurveFittingModels.quadraticCumulativeCurveFitter function."""

    Rsq = []
    RsqAdj = []
    Fstat = []
    p_Fstat = []
    Log_l = []
    AIC = []
    BIC = []

    x_list = []
    y_list = []
    results_list = []

    n_RAU = len(ll_value_db)   # number of RAUs
    estparams = np.zeros((n_RAU, 3))
    for i in xrange(len(ll_value_db)):
        val_dict = ll_value_db[i]
        cost = val_dict['cost']
        cost.insert(0, 0)
        obj = val_dict['agg_obj']
        obj.insert(0, 0)
        x = np.array(cost)
        y = np.array(obj)
        X = sm.add_constant(np.column_stack((x, x**2)))

        model = sm.OLS(y, X)
        results = model.fit()
        estparams[i, :] = results.params

        x_list.append(x)
        y_list.append(y)
        results_list.append(results)
        Rsq.append(results.rsquared)
        RsqAdj.append(results.rsquared_adj)
        Fstat.append(results.fvalue)
        p_Fstat.append(results.f_pvalue)
        Log_l.append(results.llf)
        AIC.append(results.aic)
        BIC.append(results.bic)
    filedict = {'RAU': rau_list,
                'R_squared': Rsq,
                'R_squared_adj': RsqAdj,
                'F_statistic': Fstat,
                'F_statistic_prob': p_Fstat,
                'Log-likelihood': Log_l,
                'AIC': AIC,
                'BIC': BIC}
    df = pandas.DataFrame(filedict)
    df = df[['RAU', 'R_squared', 'R_squared_adj', 'F_statistic',
             'F_statistic_prob', 'Log-likelihood', 'AIC', 'BIC']]
    df.set_index(['RAU'], inplace=True)
    df.to_csv(os.path.join(pdict[u'intermediate'],
              'rau_regression_summary.csv'))

    # save_as = os.path.join(pdict[u'intermediate'],
    # 'll_regressions.png')
    # plot_ll_curve(rau_list, x_list, y_list, save_as, results_list)
    return estparams


def plot_ll_curve(rau_list, x_list, y_list, save_as, results_list=None):
    """Plot data and regression from lower level optimization."""

    for i in xrange(len(x_list)):
        plt.plot(x_list[i], y_list[i], 'o',
                 label='RAU %d' % int(rau_list[i]))
        if results_list is not None:
            plt.plot(x_list[i], results_list[i], "^")
    if results_list is not None:
        plt.plot(x_list[0], results_list[0], "^", label='OLS')
    plt.xlabel('Cost')
    plt.ylabel('Aggregate objective')
    plt.legend(loc='best')
    plt.savefig(save_as, bbox_inches='tight')
    plt.close()


def bilevel_optimization(pdict, objective_list, model_list, intervention_list,
                         cost_list, mask_type, budget, row=0,
                         evaluate_solution=True, save_solution=True,
                         results_suffix=None, tables_list=None,
                         use_existing=False):
    """The most general 'outer-level' function to call most of the
    functionality of the marginal value script.  This function runs the hybrid
    bi-level optimization for one set of factor weights.
    It produces as output a summary table of results containing predicted and
    realized objective returns for each RAU for each factor weight combination,
    and a table of diagnostics."""

    use_existing = True  # speed up for testing!

    diagnostic_dict = {'table_row': [], 'RAU': [], 'budget_slack': [],
                       'num_IUs': [], 'num_IU_converted': []}

    # set up summary results dictionary
    summary_results = {
        'table_row': [],
        'RAU': [],
        'solution': [],
        'predicted_agg_obj_ul': [],
        'predicted_agg_obj_ll': [],
        }
    for objective in objective_list:
        summary_results['predicted_' + objective.name] = []
        summary_results[objective.name + '_weight'] = []

    if evaluate_solution:
        save_solution = True
        summary_results['realized_agg_obj'] = []
        for objective in objective_list:
            summary_results['realized_' + objective.name] = []

    # set up directory structure
    for idx in xrange(len(model_list)):
        model_list[idx].args[u'lulc_uri'] = pdict[u'lulc']
        model_list[idx].args[u'biophysical_table_uri'] = pdict[u'biophys']

    intermediate_dir = os.path.join(pdict[u'outerdir'], 'intermediate')
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    pdict[u'intermediate'] = intermediate_dir

    output_dir = os.path.join(pdict[u'outerdir'], 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pdict[u'output'] = output_dir

    sol_map_dir = os.path.join(output_dir, 'solution_maps')
    if not os.path.exists(sol_map_dir):
        os.makedirs(sol_map_dir)
    pdict[u'sol_map_dir'] = sol_map_dir

    identify_undefined_pixels(pdict, intervention_list)
    # solve optimization problem
    rau_list, ll_value_db = get_solution_db(pdict, objective_list,
                                            model_list, cost_list,
                                            intervention_list, mask_type,
                                            use_existing,
                                            tables_list=tables_list)
    for idx in xrange(len(ll_value_db)):
        value_db_RAU = ll_value_db[idx]
        warn_convex, warn_monot = check_curve(value_db_RAU)
        if warn_convex:
            # curve is not convex
            print 'lower level curve for RAU %d is not convex' \
                   % rau_list[idx]
        if warn_monot:
            # curve is not monotonic
            print 'lower level curve for RAU %d is not monotonic' \
                   % rau_list[idx]
    max_cost = [max(rau_dict['cost']) for rau_dict in ll_value_db]
    # save lower-level value databases to file
    for idx in xrange(len(rau_list)):
        rau = int(rau_list[idx])
        df = pandas.DataFrame(ll_value_db[idx])
        filename = 'solution_curve_rau%d_row%d.csv' % (rau, row)
        df.to_csv(os.path.join(pdict[u'output'], filename))

    print "..... fitting upper-level curve ..... "
    params = michaelis_menten_fitter(pdict, row, rau_list, ll_value_db,
                                     results_suffix)
    optx, ypred = michaelis_menten_solver(params, max_cost, budget)

    if optx is None:
        soln_margv = {}
    else:
        value_db = retrieve_implementation_solution(
                optx, pdict, objective_list, model_list, intervention_list,
                rau_list, mask_type, use_existing, save=save_solution, row=row)
        for idx in xrange(len(value_db)):
            rau = rau_list[idx]
            diagnostic_dict['RAU'].append(rau)
            diagnostic_dict['table_row'].append(row)
            rau_dict = value_db[idx]
            diagnostic_dict['num_IUs'].append(
                rau_dict['num_IUs'][0])
            diagnostic_dict['num_IU_converted'].append(
                rau_dict['num_IU_converted'][0])
            summary_results['predicted_agg_obj_ll'].append(
                rau_dict['agg_obj'][0])
            # calculate budget slack (unspent budget)
            avail_budget = rau_dict['budget']
            cost = rau_dict['cost']
            budget_slack = [(avail_budget - cost) for avail_budget, cost in
                            zip(avail_budget, cost)]
            for item in budget_slack:
                diagnostic_dict['budget_slack'].append(item)

        if evaluate_solution:
            print "evaluating solution ......"
            soln_margv = calc_soln_agg_obj(pdict, model_list,
                                           objective_list, rau_list,
                                           intervention_list, value_db,
                                           row=row, rsuf=results_suffix)
            soln_margv['realized_cost'] = []
            for idx in xrange(len(value_db)):
                rau_dict = value_db[idx]
                soln_margv['realized_cost'].append(rau_dict['cost'][0])
        else:
            soln_margv = {}

        # collect summary results for this weight combination
        soln_margv['RAU'] = rau_list
        soln_margv['table_row'] = []
        soln_margv['solution'] = []
        soln_margv['predicted_agg_obj_ul'] = []
        for objective in objective_list:
            soln_margv[objective.name + '_weight'] = []

        for rau in rau_list:
            soln_margv['table_row'].append(row)
            for objective in objective_list:
                soln_margv[objective.name + '_weight'].append(
                           objective.weight)
        soln_margv['solution'] = optx.tolist()
        soln_margv['predicted_agg_obj_ul'] = ypred.tolist()
        for objective in objective_list:
            soln_margv['predicted_' + objective.name] = []
            for idx in xrange(len(value_db)):
                soln_margv['predicted_' + objective.name].append(
                            value_db[idx][objective.name][0])
        for key in soln_margv.keys():
            for item in soln_margv[key]:
                summary_results[key].append(item)
    return summary_results, diagnostic_dict


def margv_tables_from_csv(pdict, objective_list, folder, identifier):
    """Folder: where the csv tables are located.  Identifier: how to identify
    the folder where these should be stored (will be located in
    pdict[u'intermediate'])."""

    tables_list = []

    intermediate_dir = os.path.join(pdict[u'outerdir'], 'intermediate')
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    pdict[u'intermediate'] = intermediate_dir

    tables_folder = os.path.join(pdict[u'intermediate'], identifier)
    if not os.path.exists(tables_folder):
        os.makedirs(tables_folder)

    rau_raster = get_rau_raster(pdict)
    rau_list = pygeoprocessing.geoprocessing.unique_raster_values_uri(
                                                                    rau_raster)

    csv_files = [f for f in os.listdir(folder) if f[-4:] == '.csv']
    assert len(csv_files) > 0, "No marginal value csv files identified"
    for objective in objective_list:
        objective_files = [f for f in csv_files if re.search(
                           objective.name, f)]
        rau_list_obj = [int(re.search('rau(.+?).csv', file).group(1)) for
                        file in objective_files]
        assert rau_list_obj == rau_list, """Supplied marginal value tables must
                                            match RAUs identified by rau_shp"""
        for filename in objective_files:
            rau = int(re.search('rau(.+?).csv', file).group(1))
            file = os.path.join(folder, filename)
            array = np.genfromtxt(file, delimiter=',')
            save_as = os.path.join(tables_folder, (objective.name +
                                   "_rau%d.npy" % rau))
            np.save(save_as, array)
            tables_list.append(save_as)
    return tables_list


def npy_to_csv(folder):
    csv_dir = os.path.join(folder, 'csv')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    npy_files = [f for f in os.listdir(folder) if re.search('.npy', f)]
    for file in npy_files:
        arr = np.load(os.path.join(folder, file))
        csv_name = os.path.join(csv_dir, file[:-4] + '.csv')
        np.savetxt(csv_name, arr, delimiter=",")


def is_none(value):
    """Convert a string indicating 'None' or missing value to None."""

    if value == 'NA' or value == 'None':
        return None
    else:
        return value


def run_from_exp_table(table):
    """Launch the routine with inputs from a csv file."""

    exp_df = pandas.read_table(table, sep=',')
    outerdir_list = exp_df['outerdir'].tolist()
    assert len(set(outerdir_list)) == 1, """All rows within one experimental
                                table must share outer directory (outerdir)"""
    obj_num_list = [int(re.search('obj(.+?)_name', n).group(1)) for n in
                    exp_df.columns.values if re.search('obj(.+?)_name', n)]
    model_num_list = [int(re.search('mod(.+?)_name', n).group(1)) for n in
                      exp_df.columns.values if re.search('mod(.+?)_name', n)]
    for row in xrange(len(exp_df)):
        pdict = {
            u'outerdir': exp_df.iloc[row].outerdir,
            u'rau_shp': exp_df.iloc[row].rau_shp,
            u'lulc': exp_df.iloc[row].lulc,
            u'biophys': exp_df.iloc[row].biophysical_table,
        }
        mask_type = exp_df.iloc[row].mask_type
        intervention_list = exp_df.iloc[row].intervention_list.split(';')
        cost_str = exp_df.iloc[row].cost_list.split(';')
        cost_list = [float(str) for str in cost_str]
        budget = float(exp_df.iloc[row].budget)
        evaluate_solution = exp_df.iloc[row].evaluate_sol
        save_solution = exp_df.iloc[row].save_sol
        results_suffix = is_none(exp_df.iloc[row].results_suffix)

        model_list = []
        for midx in model_num_list:
            name = exp_df.iloc[row]['mod%d_name' % midx]
            args_table = exp_df.iloc[row]['mod%d_args_file' % midx]
            with open(args_table, 'r') as inf:
                model_args = eval(inf.read())
            module = is_none(exp_df.iloc[row]['mod%d_module' % midx])
            output = exp_df.iloc[row]['mod%d_output' % midx]
            evaluate = exp_df.iloc[row]['mod%d_eval' % midx]
            # TODO this 'raster location' idea is awkward
            # but it should work
            if ('obj%d_raster_location' % midx) in exp_df.columns.values:
                if exp_df.iloc[row]['obj%d_raster_location' % midx] != 'None':
                    model_args[u'workspace_dir'] = exp_df.iloc[row][
                                                'obj%d_raster_location' % midx]
            model = InVEST_model(name, model_args, module, output, evaluate)
            model_list.append(model)
            if ('obj%d_raster_location' % midx) in exp_df.columns.values:
                if is_none('obj%d_raster_location' % midx) is not None:
                    get_ll_data_from_rasters(pdict, model, intervention_list,
                                             mask_type)
        objective_list = []
        for oidx in obj_num_list:
            name = exp_df.iloc[row]['obj%d_name' % oidx]
            weight = exp_df.iloc[row]['obj%d_weight' % oidx]
            l_target = is_none(exp_df.iloc[row]['obj%d_ll_target' % oidx])
            u_target = is_none(exp_df.iloc[row]['obj%d_ul_target' % oidx])
            target_type = is_none(exp_df.iloc[row]['obj%d_target_type' % oidx])
            maximize = exp_df.iloc[row]['obj%d_maximize' % oidx]
            objective_list.append(Objective(name, weight, l_target,
                                            u_target, target_type, maximize))
        sum_weight = 0
        for objective in objective_list:
            sum_weight += objective.weight
        assert sum_weight == 1, "Objective weights must sum to 1"

        model_names = [model.name for model in model_list]
        obj_names = [objective.name for objective in objective_list]
        assert len(set(model_names) - set(obj_names)) == 0, """Each model name
                                                must match an objective name"""

        sum_dict, diag_dict = bilevel_optimization(
                                 pdict, objective_list, model_list,
                                 intervention_list, cost_list, mask_type,
                                 budget, row, evaluate_solution, save_solution,
                                 results_suffix)
        if row == 0:
            summary_results = sum_dict
            diagnostic_dict = diag_dict
        else:
            for key in sum_dict:
                summary_results[key] += sum_dict[key]
            for key in diagnostic_dict:
                diagnostic_dict[key] += diag_dict[key]

    sum_df = pandas.DataFrame(summary_results)
    sum_df.set_index(['table_row'], inplace=True)
    diagnostic_df = pandas.DataFrame(diagnostic_dict)

    diagnostic_df.to_csv(os.path.join(pdict[u'output'], 'diagnostics.csv'))
    sum_df.to_csv(os.path.join(pdict[u'output'], 'summary_results.csv'))
