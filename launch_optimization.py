"""Integer optimization of livestock and water services."""
import os
import shutil

from osgeo import gdal
import re
import pandas
import numpy as np

import pygeoprocessing.geoprocessing
import marginal_value as mv


def integer_optim_Peru(objective_list, suf):
    """Calculate optimal intervention portfolio for a set of objective weights.

    Parameters:
        objective_list (list): list of objective objects containing info about
            each objective used to construct the optimization problem including
            objective weight, target, and whether it should be maximized or not
        suf (string): results suffix that will be appended to the filename of
            solution

    Side effects:
        creates or modifies a csv file containing the solution, the optimal
            intervention set
        creates or modifies a csv file containing scores, objective scores for
            the optimal intervention set

    Returns:
        solution_filename, path to csv file where solution was saved

    """
    intervention_list = [
        'camelid_high', 'camelid_high_rot', 'camelid_low', 'camelid_low_rot',
        'cow_high', 'cow_high_rot', 'cow_low', 'cow_lot_rot',
        'sheep_high', 'sheep_high_rot', 'sheep_low', 'sheep_low_rot']
    pdict = {
        u'outerdir': u"C:/Users/Ginger/Dropbox/NatCap_backup/CGIAR/Peru/InVEST_optimization_results/11.23.16/animal_weights_survey_default_beta",
        u'rau_shp': u"C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Peru/boundaries/canete_basin.shp",
        u'lulc': u"C:/Users/Ginger/Documents/NatCap/GIS_local/CGIAR/Peru/Land_Use/Final_cobertura_Canete.tif",
        }
    intermediate_dir = os.path.join(pdict[u'outerdir'], 'intermediate')
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    pdict[u'intermediate'] = intermediate_dir

    output_dir = os.path.join(pdict[u'outerdir'], 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pdict[u'output'] = output_dir
    rau = 0
    csv_folder = os.path.join(pdict['outerdir'], "marginal_value_csvs")
    tables_folder = "integer_optimizer_data"

    def generate_ll_input_data():
        tables_list = mv.margv_tables_from_csv(
            pdict, objective_list, csv_folder, tables_folder)
        # move marginal value tables just generated
        for obj in objective_list:
            int_folder = os.path.join(intermediate_dir, 'rau_%s' % obj.name)
            if not os.path.exists(int_folder):
                os.makedirs(int_folder)
            copyfrom = os.path.join(
                intermediate_dir, tables_folder, '%s_rau0.npy' % obj.name)
            copyto = os.path.join(int_folder, '%s_rau0.npy' % obj.name)
            shutil.copyfile(copyfrom, copyto)
            copyto = os.path.join(int_folder, 'rau0.npy')
            shutil.copyfile(copyfrom, copyto)

        # normalize values
        for objective in objective_list:
            if objective.name == 'cost' or objective.name == 'Cost':
                continue
            folder = os.path.join(
                pdict[u'intermediate'], 'rau_' + objective.name)
            mv.normalize_values(folder, objective.maximize)

        # rename normalized arrays
        for obj in objective_list:
            copyfrom = os.path.join(
                pdict[u'intermediate'], 'rau_%s' % obj.name, 'norm',
                '%s_rau0.npy' % obj.name)
            copyto = os.path.join(
                pdict[u'intermediate'], 'rau_%s' % obj.name, 'norm',
                'rau0.npy')
            shutil.move(copyfrom, copyto)
    generate_ll_input_data()
    ll_problem = {
        'weights': {},
        'targets': {},
        'targettypes': {}}
    for objective in objective_list:
        ll_problem['weights'][objective.name] = objective.weight
        if objective.l_target is not None:
            ll_problem['targets'][objective.name] = objective.l_target
        if objective.target_type is not None:
            ll_problem['targettypes'][objective.name] = objective.target_type

    ll_data = {'factornames': []}
    for objective in objective_list:
        ll_data['factornames'].append(objective.name)
        if objective.name == 'cost' or objective.name == 'Cost':
            rau_dir = os.path.join(
                pdict[u'intermediate'], 'rau_' + objective.name)
        else:
            rau_dir = os.path.join(
                pdict[u'intermediate'], 'rau_' + objective.name, 'norm')
        file = os.path.join(rau_dir, 'rau' + str(rau) + '.npy')
        if not os.path.isfile(file):
            raise Exception("file %s not found" % file)
        ll_data[objective.name] = np.load(file)

    # get un-normalized objective data
    ll_marg_data = ll_data.copy()
    for objective in objective_list:
        rau_dir = os.path.join(
            pdict[u'intermediate'], 'rau_' + objective.name)
        file = os.path.join(rau_dir, 'rau' + str(rau) + '.npy')
        if not os.path.isfile(file):
            raise Exception("file %s not found" % file)
        ll_marg_data[objective.name] = np.load(file)

    solution, scores = mv.integer_optimization(
        ll_data, ll_problem, rau, marg_data=ll_marg_data,
        tiebreaker_intervention=0)
    solution_filename = os.path.join(
        pdict['outerdir'], 'output', 'solution%s.csv' % suf)
    scores_filename = os.path.join(
        pdict['outerdir'], 'output', 'scores%s.csv' % suf)
    solution_df = pandas.DataFrame({'solution': solution})
    solution_df.to_csv(solution_filename)
    scores_df = pandas.DataFrame(scores, index=[0])
    scores_df.to_csv(scores_filename)
    return solution_filename


def translate_soln_to_lulc(solution_table, out_name):
    """Generate landcover raster from one optimal solution.

    Parameters:
        solution_table (string): path to csv table containing optimal
            intervention portfolio according to one set of objective weights
        out_name (string): file location where landcover raster should be
            saved

    Side effects:
        creates or modifies a geotiff at the location `out_name`

    Returns:
        None

    """
    hru_lulc_table = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\hru_definition_table.csv"
    HRU_raster = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Peru\Other_spatial_data\HRU_all.tif"
    HRU_codes = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Peru\summarized_by_zone\HRU_codes_11.8.16.csv"

    sol_df = pandas.read_csv(solution_table)
    HRU_df = pandas.read_csv(HRU_codes)
    sol_joined = pandas.concat([sol_df, HRU_df], axis=1)

    out_datatype = gdal.GDT_Int32
    source_dataset = gdal.Open(HRU_raster)
    band = source_dataset.GetRasterBand(1)
    out_nodata = band.GetNoDataValue()

    lulc_df = pandas.read_csv(hru_lulc_table)
    merged_df = pandas.merge(sol_joined, lulc_df, on="HRU", how="outer")
    merged_df['soln_int'] = merged_df['solution'].astype(float)
    merged_df['sb_lu'] = merged_df['sb_lu'].astype(float)
    merged_df.loc[
        merged_df['solution'].notnull(), 'new_lulc'] = merged_df[
            'sb_lu'] * 100 + merged_df['soln_int']
    merged_df.loc[merged_df['solution'].isnull(), 'new_lulc'] = merged_df[
        'sb_lu']

    value_map = {row[3]: row[9] for row in merged_df.itertuples()}
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        HRU_raster, value_map, out_name, out_datatype, out_nodata)

    band = None
    del source_dataset


def translate_solution(solution_csv, HRU_codes, HRU_raster, raster_out_uri):
    """Create raster showing optimal intervention for each HRU.

    Parameters:
        solution_csv (string): path to csv file containing the optimal
            intervention portfolio according to one set of objective weights
        HRU_codes (string): path to csv file containing HRU codes in a single
            column. This file can be created by copying the 'zone' column from
            one of the marginal value rasters (e.g.,
            livestock_mv_by_HRU_9.29.16.csv).
        HRU_raster (string): path to geotiff containing hydrologic response
            units (HRUs) indexed to the integer codes in `HRU_codes`
        raster_out_uri (string): path to location on disk where optimal
            intervention geotiff should be saved

    Side effects:
        Creates or modifies the geotiff located at `raster_out_uri`

    Returns:
        None

    """
    HRU_list = pygeoprocessing.geoprocessing.unique_raster_values_uri(
        HRU_raster)
    sol_df = pandas.read_csv(solution_csv)
    HRU_df = pandas.read_csv(HRU_codes)
    assert len(set(HRU_list) - set(HRU_df.HRU)) == 0, """Error: HRU raster does
                                                        not match HRU codes"""
    sol_joined = pandas.concat([sol_df, HRU_df], axis=1)

    out_datatype = 3
    source_dataset = gdal.Open(HRU_raster)
    band = source_dataset.GetRasterBand(1)
    out_nodata = band.GetNoDataValue()

    value_map = {row[3]: row[2] for row in sol_joined.itertuples()}
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
               HRU_raster, value_map, raster_out_uri, out_datatype, out_nodata)

    band = None
    del source_dataset


def integer_optim_wrapper():
    """Run integer optimization at series of objective weight combinations.

    This function calls `integer_optim_Peru`, `translate_solution`, and
    `translate_soln_to_lulc` to calculate optimal interventions for a series of
    objective weights and generate tables and maps from that solution.

    Side effects:
        creates or modifies files located at hard-coded locations on disk

    Returns:
        None

    """
    weight_range = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for livestock_weight in weight_range:
        for sdr_weight in weight_range:
            for swy_weight in weight_range:
                if (livestock_weight == sdr_weight and
                        sdr_weight == swy_weight):
                    if swy_weight < 1:  # only run once with equal weights
                        continue
                if (livestock_weight == 0 and sdr_weight == 0 and
                        swy_weight == 0):
                    continue
                sed_obj = mv.Objective(
                    'sdr', sdr_weight, None, None, None, maximize=False)
                swy_obj = mv.Objective(
                    'swy', swy_weight, None, None, None, maximize=True)
                livestock_obj = mv.Objective(
                    'livestock', livestock_weight, None, None, None,
                    maximize=True)
                objective_list = [sed_obj, swy_obj, livestock_obj]
                suf = 'livestock_{}_sdr_{}_swy_{}'.format(
                    livestock_obj.weight, sed_obj.weight, swy_obj.weight)
                raster_out_uri = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\InVEST_optimization_results\11.23.16\animal_weights_survey_default_beta\output\solution_map%s.tif" % suf
                if not os.path.exists(raster_out_uri):
                    solution_csv = integer_optim_Peru(objective_list, suf)
                    HRU_codes = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Peru\summarized_by_zone\HRU_codes_11.8.16.csv"
                    HRU_raster = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Peru\Other_spatial_data\HRU_priority_FESC_RYEG.tif"
                    translate_solution(solution_csv, HRU_codes, HRU_raster, raster_out_uri)
                    lulc_out_name = r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\InVEST_optimization_results\11.23.16\animal_weights_survey_default_beta\output\solution_lulc%s.tif" % suf
                    translate_soln_to_lulc(solution_csv, lulc_out_name)


def collate_scores(output_folder, save_as):
    """Collect scores into a file for plotting frontiers.

    Parameters:
        output_folder (string): path to local file folder containing a series
            of csv tables, one for each objective weight combination,
            indicating objective scores for the optimal solution
        save_as (string): path to location where summary of objective scores
            across objective weights should be saved

    Side effects:
        creates or modifies the csv table indicated by the path `save_as`

    Returns:
        None

    """
    scores_files = [
        f for f in os.listdir(output_folder) if f.startswith('scores')]
    f = scores_files[0]
    sol_df = pandas.read_csv(os.path.join(output_folder, f))
    objectives = sol_df.columns.values.tolist()
    objectives.remove('objective')
    objectives.remove('Unnamed: 0')

    sum_dict = {}
    for obj in objectives:
        sum_dict['{}_weight'.format(obj)] = []
        sum_dict['{}_score'.format(obj)] = []
    for f in scores_files:
        sol_df = pandas.read_csv(os.path.join(output_folder, f))
        for obj in objectives:
            score = sol_df.get_value(0, obj)
            try:
                pattern = '{}_(.+?)_'.format(obj)
                weight = re.search(pattern, f).group(1)
            except IndexError:
                pattern = '{}_(.+?).csv'.format(obj)
                weight = re.search(pattern, f).group(1)
            sum_dict['{}_weight'.format(obj)].append(weight)
            sum_dict['{}_score'.format(obj)].append(score)
    sum_df = pandas.DataFrame(sum_dict)
    sum_df.to_csv(save_as)


def collate_solutions(output_folder, objective_list):
    """Collect solutions from several portfolios.

    Parameters:
        output_folder (string): path to directory on disk that contains
            solution summaries
        objective_list (list): list of strings identifying the order of
            objectives

    Side effects:
        creates or modifies the following files in `output_folder`:
            'solution_summary.csv'
            'solution_index.csv'

    Returns:
        None

    """
    solutions_files = [
        f for f in os.listdir(output_folder) if f.startswith('solution') and
        f.endswith('.csv')]
    df_list = []
    weight_dict = {obj: [] for obj in objective_list}
    weight_dict['soln_index'] = []
    for idx in xrange(len(solutions_files)):
        f = solutions_files[idx]
        weight_dict['soln_index'].append(idx)
        for obj in objective_list:
            try:
                pattern = '{}_(.+?)_'.format(obj)
                weight = re.search(pattern, f).group(1)
            except IndexError:
                pattern = '{}_(.+?).csv'.format(obj)
                weight = re.search(pattern, f).group(1)
            weight_dict[obj].append(weight)
        df = pandas.read_csv(os.path.join(output_folder, f))
        del df['Unnamed: 0']
        df.columns = [idx]
        df_list.append(df)
    result_df = pandas.concat(df_list, axis=1)
    weight_df = pandas.DataFrame(weight_dict)
    result_df.to_csv(os.path.join(output_folder, "solution_summary.csv"),
                     index=False)
    weight_df.to_csv(os.path.join(output_folder, "solution_index.csv"),
                     index=False)


def solution_agreement(HRU_codes, solution_summary, save_as):
    """Calculate agreement metrics from a set of portfolios.

    Parameters:
        HRU_codes (string): path to csv file containing HRU codes in a single
            column of integers
        solution_summary (string): path to csv file containing summary of
            solutions among objective weight combinations
        save_as (string): path to location on disk where agreement metrics
            should be saved

    Side effects:
        creates or modifies the csv file indicated by `save_as`

    Returns:
        None

    """
    sol_df = pandas.read_csv(solution_summary)
    stat_df = pandas.read_csv(HRU_codes)

    # proportion of runs where each HRU was selected for intervention
    stat_df['prop_selected'] = (
        sol_df.astype(bool).sum(axis=1) / sol_df.shape[1])

    # most often chosen intervention: mode
    stat_df['mode'] = stats.mode(sol_df, axis=1)[0]

    stat_df.set_index("HRU")
    stat_df.to_csv(save_as, index=False)


def create_agreement_rasters(agreement_summary, HRU_raster, raster_out_dir):
    """Make rasters of agreement metrics.

    Parameters:
        agreement_summary (string): path to csv containing portfolio agreement
            metrics
        HRU_raster (string): path to geotiff containing HRUs
        raster_out_dir (string): path to location on disk where agreement
            rasters should be saved

    Side effects:
        creates or modifies the following files in `raster_out_dir`:
            'proportion_selected.tif'
            '"most_often_chosen.tif'

    Returns:
        None

    """
    HRU_list = pygeoprocessing.geoprocessing.unique_raster_values_uri(
        HRU_raster)
    agreement_df = pandas.read_csv(agreement_summary)
    assert len(set(HRU_list) - set(agreement_df.HRU)) == 0, """Error: HRU
        raster does not match HRU codes"""

    source_dataset = gdal.Open(HRU_raster)
    band = source_dataset.GetRasterBand(1)
    out_nodata = band.GetNoDataValue()

    # proportion selected
    out_datatype = gdal.GDT_Float32
    raster_out_uri = os.path.join(raster_out_dir, "proportion_selected.tif")
    value_map = {row[1]: row[2] for row in agreement_df.itertuples()}
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        HRU_raster, value_map, raster_out_uri, out_datatype, out_nodata)

    # (modal) most often chosen intervention
    out_datatype = gdal.GDT_Int32
    raster_out_uri = os.path.join(raster_out_dir, "most_often_chosen.tif")
    value_map = {row[1]: row[4] for row in agreement_df.itertuples()}
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        HRU_raster, value_map, raster_out_uri, out_datatype, out_nodata)

    band = None
    del source_dataset


if __name__ == "__main__":
    # Calculate optimal intervention portfolios at range of objective weights
    integer_optim_wrapper()
    folder = 'animal_weights_survey_default_beta'
    output_folder = os.path.join(
        r"C:\Users\Ginger\Dropbox\NatCap_backup\CGIAR\Peru\InVEST_optimization_results\11.23.16",
        folder, "output")
    save_as = os.path.join(
        output_folder, "{}_scores_summary.csv".format(folder))
    collate_scores(output_folder, save_as)
    objective_list = ['livestock', 'sdr', 'swy']
    collate_solutions(output_folder, objective_list)
    solution_summary = os.path.join(output_folder, "solution_summary.csv")

    # Summarize portfolio agreement among objective weight combinations
    agreement_summary = os.path.join(output_folder, 'solution_agreement_summary.csv')
    solution_agreement(HRU_codes, solution_summary, agreement_summary)
    HRU_raster = r"C:\Users\Ginger\Documents\NatCap\GIS_local\CGIAR\Peru\Other_spatial_data\HRU_priority_FESC_RYEG.tif"
    raster_out_dir = os.path.join(output_folder, 'agreement_rasters')
    create_agreement_rasters(agreement_summary, HRU_raster, raster_out_dir)
