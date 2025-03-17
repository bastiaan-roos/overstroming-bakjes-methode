"""
Combineer de resultaten van de schadeberekeningen in een overzichtelijk bestand.
"""

import pandas as pd
import geopandas as gpd
from osgeo import ogr

from bakjes_overstroming.paden import *

ogr.UseExceptions()


def combine_schade_results():

    # alle berekeningen
    calculations = gpd.read_file(gpkg_file_path, layer=drempelwaarde_layer_name)

    out = calculations[['cluster', 'boezem', 'drempel_og_1', 'drempel_boezem_1',
                        'evenw_met_drempel_open', 'evenw_met_drempel_gesloten',
                        'evenw_met_drempel_open_achter', 'evenw_met_drempel_gesloten_achter',
                        'grid_volume']]
    out = out.rename(columns={'cluster': 'og_id', 'boezem': 'boezem_id'})
    out['og_id'] = out['og_id'].astype(int)
    out['boezem_id'] = out['boezem_id'].astype(int)

    # overstromingsgebieden
    overstromingsgebieden = gpd.read_file(gpkg_file_path, layer=overstromingsgebieden_layer_name)
    overstromingsgebieden = overstromingsgebieden[['cluster_int', 'ogname']]
    overstromingsgebieden = overstromingsgebieden.rename(columns={'cluster_int': 'og_id', 'ogname': 'og_naam'})
    # overstromingsgebieden['og_id'] = overstromingsgebieden['og_id'].astype(int)

    # boezemgebieden
    boezemgebieden = gpd.read_file(gpkg_file_path, layer=watergebieden_opp_layer_name)
    boezemgebieden = boezemgebieden[['id', 'name']]
    boezemgebieden = boezemgebieden.rename(columns={'id': 'boezem_id', 'name': 'boezem_naam'})
    boezemgebieden['boezem_id'] = boezemgebieden['boezem_id'].astype(int)
    # boezemgebieden.set_index('boezem', inplace=True)

    out = out.merge(overstromingsgebieden, left_on='og_id', right_on='og_id', how='left')
    out = out.merge(boezemgebieden, left_on='boezem_id', right_on='boezem_id', how='left')
    # change order
    out = out[['og_id', 'og_naam', 'boezem_id', 'boezem_naam', 'drempel_og_1', 'drempel_boezem_1',
                        'evenw_met_drempel_open', 'evenw_met_drempel_gesloten',
                        'evenw_met_drempel_open_achter', 'evenw_met_drempel_gesloten_achter',
                        'grid_volume']]
    # round all to 2 decimals
    out = out.round(2)

    # add results
    # results in excel file to read in pandas are in range A3:B50  (row 3 is the header)
    batch = pd.read_csv(batch_file)

    results = []

    results_max = {}

    results_cluster = {}

    for index, row in batch.iterrows():
        inundatie_file = row['inundatie_file']
        result_file = os.path.join(script_dir, 'results', f'{inundatie_file[:-4]}', 'schades.xls')
        if not os.path.exists(result_file):
            print(f"skipping {inundatie_file}")
            continue
        df = pd.read_excel(result_file, header=2)

        # transpose the dataframe, make column A the header and column b the values
        dft = df.T
        dft.columns = dft.iloc[0]
        # only keep first line
        dft = dft[1:2]
        og_id = int(row['cluster']) if not pd.isna(row['cluster']) else None
        boezem_id = int(row['boezem']) if not pd.isna(row['boezem']) else None

        dft['og_id'] = og_id
        dft['boezem_id'] = boezem_id
        dft['waterniveau'] = row['waterniveau']
        dft['comp_keringen'] = row['comp_keringen']
        dft['drempel'] = row['drempel']
        results.append(dft)

        if row['waterniveau'] == 'max':
            results_max[og_id] = dft['Totaal'].values[0]
            continue

        if og_id not in results_cluster:
            results_cluster[og_id] = {}

        if boezem_id not in results_cluster[og_id]:
            results_cluster[og_id][boezem_id] = {
            }
        total = dft['Totaal'].values[0]
        results_cluster[og_id][boezem_id][f"{row['comp_keringen']}_{row['drempel']}"] = total

    cluster_out = []
    for cluster, boezems in results_cluster.items():
        for boezem, damages in boezems.items():
            damages['og_id'] = int(cluster)
            damages['boezem_id'] = int(boezem)
            damages['max'] = results_max.get(cluster, None)
            cluster_out.append(damages)

    cluster_out = pd.DataFrame(cluster_out)
    # join to out
    out = out.merge(cluster_out, left_on=['og_id', 'boezem_id'], right_on=['og_id', 'boezem_id'], how='left')
    out = out.sort_values(['og_id', 'boezem_id'])
    out.to_csv(os.path.join(script_dir, 'results', 'schades_per_og_boezem.csv'))

    dfb = pd.DataFrame(
        results_cluster.values()
    )
    dfb.to_csv(os.path.join(script_dir, 'results', 'schades_per_oc.csv'))


    # save sub schade bedragen per scenario
    df = pd.concat(results)
    df = df.merge(boezemgebieden, left_on='boezem_id', right_on='boezem_id', how='left')
    df = df.merge(overstromingsgebieden, left_on='og_id', right_on='og_id', how='left')

    # move columns cluster, boezem and waterniveau to the front
    first_cols = ['og_id', 'og_naam', 'boezem_id', 'boezem_naam','waterniveau', 'comp_keringen', 'drempel']
    df = df[first_cols + [col for col in df.columns if col not in first_cols]]
    df.to_csv(os.path.join(script_dir, 'results', 'schades_totaal.csv'))


if __name__ == '__main__':
    combine_schade_results()
