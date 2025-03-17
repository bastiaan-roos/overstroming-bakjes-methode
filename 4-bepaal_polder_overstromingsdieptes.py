"""
Bepalen van de overstromingsdieptes
Input is een gecombineerd hoogtegrid (tif), geopackage met nodige gegevens (clusters, boezemgebieden, etc), en
bergings-diepte curve van de boezemcompartimenten
"""


from osgeo import ogr, gdal

from osgeo_utils import gdal_merge
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
import numpy as np


from bakjes_overstroming.paden import *

ogr.UseExceptions()


scenarios = [
    {
        'name': 'gesloten',
        'volume_curve_template': 'total_curve_{}.csv',
    }, {
        'name': 'open',
        'volume_curve_template': 'total_curve_open_{}.csv',
    }
]

drempel_percentiles = [0, 1, 5]
buffer_distance_drempel = 80

# percentiles = range(0, 101, 1)


def interpolate_ahn(grid_file_path, output_path, maxSearchDist=150):
    """Interpoleer de gaten dicht in een grid
    """

    grid = gdal.Open(grid_file_path)

    # fill nodata gaps
    mem_raster = gdal.GetDriverByName('MEM').CreateCopy('', grid)

    mem_band = mem_raster.GetRasterBand(1)

    result = gdal.FillNodata(targetBand=mem_band, maskBand=None,
                             maxSearchDist=maxSearchDist, smoothingIterations=9)
    mem_band.FlushCache()

    # export to tiff
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(output_path, mem_raster)


def merge_aanafvoergebieden_to_cluster():
    """Combineer de aanenafvoergebieden naar overstromingsgebieden. Aan te raden om het resultaat te controleren
    en waar nodig nog te 'cleanen'"""
    # cluster aanafvoergebieden
    aanafvoergebieden = gpd.read_file(gpkg_file_path, layer=aanafvoergebieden_layer_name)
    aanafvoergebieden['cluster'] = aanafvoergebieden[aanafvoergebieden_clustering_column].astype(str)
    aanafvoergebieden = aanafvoergebieden.dissolve(by='cluster')
    aanafvoergebieden.to_file(gpkg_file_path, layer=overstromingsgebieden_layer_name, driver='GPKG')


def make_raster_for_each_cluster():
    """Maakt voor elke overstromingsgebied een hoogte grid
    """
    overstromingsgebieden = gpd.read_file(gpkg_file_path, layer=overstromingsgebieden_layer_name)

    with rasterio.open(merged_interpolated_grid_file) as src:
        raster_meta = src.meta

        for index, cluster in overstromingsgebieden.iterrows():
            print(f'processing cluster {cluster['cluster_int']}')
            geometry = cluster['geometry']
            # clip grids on areas
            clipped_grid, out_transform = mask(src, [geometry], crop=True)

            clipped_grid_meta = raster_meta.copy()
            clipped_grid_meta.update({"driver": "GTiff",
                                      "height": clipped_grid.shape[1],
                                      "width": clipped_grid.shape[2],
                                      "transform": out_transform})
            with rasterio.open(os.path.join(dest_dir_clipped, 'cluster_{0}.tif'.format(cluster['cluster_int'])), 'w', **clipped_grid_meta) as dst:
                dst.write(clipped_grid)


# def make_maaiveldcurve_for_each_cluster():
#
#     overstromingsgebieden = gpd.read_file(gpkg_file_path, layer=overstromingsgebieden_layer_name)
#
#     percentiles = range(0, 101, 1)
#     rows = []
#
#     for index, cluster in overstromingsgebieden.iterrows():
#         output = {
#             'cluster_int': cluster['cluster_int'],
#         }
#         grid_file = os.path.join(dest_dir_clipped, 'cluster_{0}.tif'.format(cluster['cluster_int']))
#         with rasterio.open(grid_file) as src:
#             raster = src.read(1)
#             raster_meta = src.meta
#
#             # grids with data
#             # set nan values to nodata
#             raster = np.where(raster == raster_meta['nodata'], np.nan, raster)
#             output['nr'] = len(raster[~np.isnan(raster)])
#             output['area'] = output['nr'] * abs(raster_meta['transform'][0] * raster_meta['transform'][4])
#
#             for percentile in percentiles:
#                 output[percentile] = np.nanpercentile(raster, percentile)
#
#             rows.append(output)
#     df = pd.DataFrame(rows)
#     df.to_csv(maaiveldcurve_filepath, index=False)
#
#
# def maak_overstromingsgebied_volume_curves():
#
#     overstromingsgebieden = gpd.read_file(gpkg_file_path, layer=overstromingsgebieden_layer_name)
#     maaiveldcurve_tot = pd.read_csv(maaiveldcurve_filepath, dtype={'cluster_int': int})
#     maaiveldcurve_tot = maaiveldcurve_tot.set_index('cluster_int')
#
#     for index, cluster in overstromingsgebieden.iterrows():
#         maaiveldcurve = maaiveldcurve_tot.loc[cluster['cluster_int']]
#
#         opp = maaiveldcurve['area']
#
#         curve = []
#         for i in percentiles:
#             curve.append({
#                 'elevation': maaiveldcurve[f'{i}'],
#                 'area': (i - 0.5) * opp / 100
#             })
#
#         df_curve = pd.DataFrame(curve)
#         df_curve['elevation'] = df_curve['elevation'] * 100
#         df_curve['elevation'] = df_curve['elevation'].round(0).astype(int)
#         df_curve = df_curve.drop_duplicates(subset='elevation', keep='first')
#         df_curve.set_index('elevation', inplace=True)
#
#         max_elevation = int(maaiveldcurve['100'] * 100)
#         min_elevation = int(maaiveldcurve['0'] * 100)
#
#         df_tot_curve = pd.DataFrame(
#             [{'elevation': el, 'area': np.nan} for el in range(min_elevation, max_elevation + 1, 1)]
#         )
#         df_tot_curve.set_index('elevation', inplace=True)
#         df_tot_curve.update(df_curve)
#         del df_curve
#         df_tot_curve.interpolate(method='linear', inplace=True)
#         df_tot_curve.bfill(inplace=True)
#         df_tot_curve['volume_cm'] = df_tot_curve['area'] * 0.01
#         df_tot_curve['volume'] = df_tot_curve['volume_cm'].cumsum()
#         df_tot_curve = df_tot_curve[(df_tot_curve.index >= -8000) & (df_tot_curve.index <= 0)]
#         df_tot_curve.to_csv(os.path.join(dest_dir, f'volume_curve_{cluster["cluster_int"]}.csv'), index=True)


def make_overstromingscurves_direct_from_grid():
    """bepaal maaivelddurve en volume curve direct vanuit grid
    """

    overstromingsgebieden = gpd.read_file(gpkg_file_path, layer=overstromingsgebieden_layer_name)

    for index, cluster in overstromingsgebieden.iterrows():
        print(f'processing cluster {cluster["cluster_int"]}')

        grid_path = os.path.join(dest_dir_clipped, f'cluster_{cluster["cluster_int"]}.tif')

        # min elevation
        raster = rasterio.open(grid_path)
        grid = raster.read(1)
        grid = np.where(grid == raster.meta['nodata'], np.nan, grid)
        min_elevation = np.nanmin(grid)
        size = abs(raster.meta['transform'][0] * raster.meta['transform'][4])

        range_from = max(int(min_elevation * 100), -800)

        out = []
        volume = 0
        for level in range(range_from, 15, 1):
            area = np.sum(grid <= (level / 100))
            area = area * size
            volume_cm = area * 0.01
            volume += volume_cm
            out.append({
                'elevation': level,
                'area': area,
                'volume_cm': volume_cm,
                'volume': volume
            })

        df = pd.DataFrame(out)
        df.set_index('elevation', inplace=True)
        df.to_csv(os.path.join(dest_dir, f'volume_curve_{cluster["cluster_int"]}.csv'), index=True)


def get_link_overstromingsgebied_boezemgebied():
    """Berekening van de drempelwaarden tussen boezem en overstromingsgebieden
    - buffer overstromingsgebieden met `buffer_distance_drempel` (=80) meter en kijk welke boezemgebieden overlappen
    - kijk vervolgens vanuit de boezem met een buffer van 80 meter wat de 0, 1 en 5
      percentiel is van de maaiveldcurve - uiteindelijk de 1% gebruikt - dit is de polder drempel
    - bereken vervolgens wat vanuit de polder gezien met een buffer van 80 meter de 1, 5, en 10 percentiel is van
      de boezemcurve - uiteindelijk 1% gekozen - dit is de boezem drempel
    """

    overstromingsgebieden = gpd.read_file(gpkg_file_path, layer=overstromingsgebieden_layer_name)
    boezemgebieden = gpd.read_file(gpkg_file_path, layer=watergebieden_opp_layer_name)

    out = []

    for index, cluster in overstromingsgebieden.iterrows():
        print(f'processing cluster {cluster["cluster_int"]}')
        geometry = cluster['geometry']
        buffer = geometry.buffer(buffer_distance_drempel)
        boezems = boezemgebieden[boezemgebieden.intersects(buffer)]
        print(boezems)
        for index_boezem, boezem in boezems.iterrows():
            print(f"get drempelwaarde polder {cluster["cluster_int"]} - boezem {boezem['id']}")
            # get maaiveldcurve polder from grid
            polder_buffer = boezem['geometry'].buffer(buffer_distance_drempel)
            polder_buffer = polder_buffer.intersection(geometry)
            # get maaiveldcurve boezem from grid
            grid = os.path.join(dest_dir_clipped, f'cluster_{cluster["cluster_int"]}.tif')
            with rasterio.open(grid) as src:
                raster, _ = mask(src, [polder_buffer], crop=True)
                raster = np.where(raster == src.meta['nodata'], np.nan, raster)
                maaiveldcurve_polder = np.nanpercentile(raster, drempel_percentiles)

            # get boezemcurve from grid
            grid = os.path.join(dest_dir_clipped_water, f'cluster_{boezem["id"]}.tif')
            with rasterio.open(grid) as src:
                raster, _ = mask(src, [buffer], crop=True)
                raster = np.where(raster == src.meta['nodata'], np.nan, raster)
                maaiveldcurve_boezem = np.nanpercentile(raster, drempel_percentiles)

            out.append({
                'cluster': int(cluster['cluster_int']),
                'geometry': polder_buffer,
                'boezem': boezem['id'],
            })
            for i, percentile in enumerate(drempel_percentiles):
                out[-1][f'drempel_og_{percentile}'] = maaiveldcurve_polder[i]
            for i, percentile in enumerate(drempel_percentiles):
                out[-1][f'drempel_boezem_{percentile}'] = maaiveldcurve_boezem[i]


    df = gpd.GeoDataFrame(out, crs=overstromingsgebieden.crs)
    df.to_file(gpkg_file_path, layer=drempelwaarde_layer_name, driver='GPKG')


def process_polder_with_achter(
        df_curve,  # with diff
        achter_settings):

    drempel = achter_settings['level']
    # get volume_curve of achterliggende polders
    og_curve_1 = pd.read_csv(os.path.join(dest_dir, f'volume_curve_{achter_settings['achter_polders_1']}.csv'), index_col='elevation')
    og_curve_2 = pd.read_csv(os.path.join(dest_dir, f'volume_curve_{achter_settings['achter_polders_2']}.csv'), index_col='elevation')

    og_curve_1['volume_og1'] = og_curve_1['volume']
    og_curve_1 = og_curve_1[['volume_og1']]

    og_curve_2['volume_og2'] = og_curve_2['volume']
    og_curve_2 = og_curve_2[['volume_og2']]

    df_curve = df_curve.merge(og_curve_1, on='elevation', how='outer')
    df_curve = df_curve.merge(og_curve_2, on='elevation', how='outer')

    df_curve['volume_og1'] = df_curve['volume_og1'].fillna(0)
    df_curve['volume_og2'] = df_curve['volume_og2'].fillna(0)

    df_curve['volume_og1_drempel'] = (og_curve_1[f'volume_og1']
                                .where(og_curve_1[f'volume_og1'].index > drempel, 0))
    df_curve['volume_og2_drempel'] = (og_curve_2[f'volume_og2']
                                .where(og_curve_2[f'volume_og2'].index > drempel, 0))

    df_curve['volume_og0'] = df_curve['volume_og']
    df_curve['volume_achter'] = df_curve['volume_og1'] + df_curve['volume_og2']
    df_curve['volume_og'] = df_curve['volume_og'] + df_curve['volume_og1_drempel'] + df_curve['volume_og2_drempel']
    df_curve['diff'] = df_curve['volume_og'] - df_curve['volume_tot']

    df_curve.sort_index(inplace=True)
    try:
        evenwichtspeil = df_curve[df_curve['volume_og'] > df_curve['volume_tot']].index[0]
        if int(evenwichtspeil) < achter_settings['level']:
            return evenwichtspeil, None

    except IndexError:
        return None, None

    volume_to_achter = df_curve.loc[evenwichtspeil, 'volume_tot'] - df_curve.loc[evenwichtspeil, 'volume_og0']
    df_curve['diff'] = df_curve['volume_achter'] - volume_to_achter
    try:
        achterliggendpeil = df_curve[df_curve['volume_achter'] > volume_to_achter].index[0]
    except IndexError:
        achterliggendpeil = None

    return evenwichtspeil, achterliggendpeil


def calc_evenwichtspeilen():
    """Bereken de evenwichtspeilen voor de overstromingsgebieden voor de verschillende scenario's en drempels en
    overstromingsgebied/ boezemcompartiment combinatie
    """

    overstromingsgebieden = gpd.read_file(gpkg_file_path, layer=overstromingsgebieden_layer_name)

    achter_settings = {}

    for index, cluster in overstromingsgebieden.iterrows():
        if np.isnan(cluster['achter_level']):
            continue

        achter_settings[cluster['cluster_int']] = {
            'level': cluster['achter_level'],
            'achter_polders_1': int(cluster['achter_polders_1']),
            'achter_polders_2': int(cluster['achter_polders_2']),
        }

    overstromingsgebieden_met_drempel = gpd.read_file(gpkg_file_path, layer=drempelwaarde_layer_name)

    overstromingsgebieden_met_drempel[f'evenw_zonder_drempel_open'] = np.nan
    overstromingsgebieden_met_drempel[f'evenw_zonder_drempel_gesloten'] = np.nan
    overstromingsgebieden_met_drempel[f'evenw_met_drempel_open'] = np.nan
    overstromingsgebieden_met_drempel[f'evenw_met_drempel_gesloten'] = np.nan

    overstromingsgebieden_met_drempel.to_file(gpkg_file_path, layer=drempelwaarde_layer_name, driver='GPKG')
    overstromingsgebieden_met_drempel = gpd.read_file(gpkg_file_path, layer=drempelwaarde_layer_name)

    for scenario in scenarios:
        for index, cluster in overstromingsgebieden_met_drempel.iterrows():
            # if not(int(cluster['cluster']) == 31 and int(cluster['boezem']) == 5):
            #     continue

            sname = scenario['name']

            og_curve = pd.read_csv(os.path.join(dest_dir, f'volume_curve_{cluster["cluster"]}.csv'), index_col='elevation')
            boezem_curve = pd.read_csv(
                os.path.join(output_path_boezem, scenario['volume_curve_template'].format(cluster["boezem"])),
                index_col='elevation'
            )
            og_curve['volume_og'] = og_curve['volume']
            og_curve = og_curve[['volume_og']]

            # link both curve and find first row where volume og_curve is higher than volume boezem_curve
            df_curve = og_curve.join(boezem_curve, lsuffix='_og', rsuffix='_boezem', how='outer')
            # order by elevation
            df_curve.sort_index(ascending=True, inplace=True)

            # keep only columns volume_og and volume_tot
            df_curve = df_curve[['volume_og', 'volume_tot']]
            df_curve['volume_og'] = df_curve['volume_og'].fillna(0)
            df_curve['volume_tot'] = df_curve['volume_tot'].ffill()

            drempel = cluster['drempel_boezem_1']
            if not np.isnan(drempel):
                df_curve.loc[df_curve.index < int(drempel * 100), 'volume_tot'] = np.nan
                df_curve['volume_tot'] = df_curve['volume_tot'].bfill()

            df_curve['diff'] = df_curve['volume_og'] - df_curve['volume_tot']
            try:
                evenwichtspeil = df_curve[df_curve['volume_og'] > df_curve['volume_tot']].index[0]
                if int(cluster['cluster']) in achter_settings:
                    evenwichtspeil, achterliggendpeil = process_polder_with_achter(
                        df_curve,
                        achter_settings[cluster['cluster']],
                    )
                    if achterliggendpeil is not None:
                        overstromingsgebieden_met_drempel.loc[index, f'evenw_zonder_drempel_{sname}_achter'] = achterliggendpeil
                    else:
                        overstromingsgebieden_met_drempel.loc[
                            index, f'evenw_zonder_drempel_{sname}_achter'] = np.nan

                print(f'evenwichtspeil {sname} {evenwichtspeil}')
                overstromingsgebieden_met_drempel.loc[index, f'evenw_zonder_drempel_{sname}'] = evenwichtspeil
            except IndexError as e:
                # niet gevonden, dus 0 mNAP
                overstromingsgebieden_met_drempel.loc[index, f'evenw_zonder_drempel_{sname}'] = 0
            # drempel = max 1% van de boezemcurve, 1% van de poldercurve
            drempel = np.nanmax([cluster['drempel_boezem_1'], cluster['drempel_og_1']])
            if not np.isnan(drempel):
                # set volume boezem onder drempel to volume at drempel
                df_curve.loc[df_curve.index < int(drempel * 100), 'volume_tot'] = np.nan

            df_curve['volume_tot'] = df_curve['volume_tot'].bfill()
            df_curve['diff'] = df_curve['volume_og'] - df_curve['volume_tot']
            df_curve.sort_index(ascending=True, inplace=True)
            try:
                evenwichtspeil = df_curve[df_curve['volume_og'] > df_curve['volume_tot']].index[0]
                if int(cluster['cluster']) in achter_settings:
                    evenwichtspeil, achterliggendpeil = process_polder_with_achter(
                        df_curve,
                        achter_settings[cluster['cluster']],
                    )
                    if achterliggendpeil is not None:
                        overstromingsgebieden_met_drempel.loc[index, f'evenw_met_drempel_{sname}_achter'] = achterliggendpeil
                    else:
                        overstromingsgebieden_met_drempel.loc[
                            index, f'evenw_met_drempel_{sname}_achter'] = np.nan

                print(f'evenwichtspeil {sname} {evenwichtspeil}')
                overstromingsgebieden_met_drempel.loc[index, f'evenw_met_drempel_{sname}'] = evenwichtspeil

            except IndexError:
                overstromingsgebieden_met_drempel.loc[index, f'evenw_met_drempel_{sname}'] = 0

    # overstromingsgebieden_met_drempel['cluster_int'] = pd.to_numeric(
    #     overstromingsgebieden_met_drempel['cluster'], downcast='integer')

    overstromingsgebieden_met_drempel.to_file(gpkg_file_path, layer=drempelwaarde_layer_name, driver='GPKG')


def get_inundatie_grids():
    """Bepaal de inundatiegrids voor de verschillende scenario's en drempels en ook het maximale, gelijk aan
    het maatgevende waterpeil (0mNAP)
    """
    # output csv file with list of inundation grids
    overstromingsgebieden = gpd.read_file(gpkg_file_path, layer=overstromingsgebieden_layer_name)

    batch = []
    achter_settings = {}

    for index, cluster in overstromingsgebieden.iterrows():

        if not np.isnan(cluster['achter_level']):

            achter_settings[cluster['cluster_int']] = {
                'level': cluster['achter_level'],
                'achter_polders_1': int(cluster['achter_polders_1']),
                'achter_polders_2': int(cluster['achter_polders_2']),
            }
        grid_file = os.path.join(dest_dir_clipped, 'cluster_{0}.tif'.format(cluster['cluster_int']))

        with rasterio.open(grid_file, 'r') as src:
            raster = src.read(1)
            raster_meta = src.meta

            # grids with data
            # set nan values to nodata
            raster = np.where(raster == raster_meta['nodata'], np.nan, 0 - raster)
            # set all values lower than 0 to nan
            raster = np.where(raster < 0, np.nan, raster)
            raster[np.isnan(raster)] = raster_meta['nodata']

            # export inundation grid
            output_file = os.path.join(inundation_dir, f'inundation_cluster_{cluster["cluster_int"]}_max_depth.tif')
            with rasterio.open(output_file, 'w', **raster_meta) as dst:
                dst.write(raster, 1)

            batch.append({
                'cluster': cluster['cluster'],
                'cluster_int': cluster['cluster_int'],
                'waterniveau': 'max',
                'peil': 0,
                'boezem': np.nan,
                'inundatie_file': os.path.basename(output_file)
            })

    # do the same voor evenwichtspeil
    overstromingsgebieden_met_drempel = gpd.read_file(gpkg_file_path, layer=drempelwaarde_layer_name)
    for index, cluster in overstromingsgebieden_met_drempel.iterrows():
        print(f'processing cluster {cluster["cluster"]}')

        grid_file = os.path.join(dest_dir_clipped, 'cluster_{0}.tif'.format(cluster['cluster']))

        with rasterio.open(grid_file, 'r') as src:
            for scenario in scenarios:
                for drempel in ['met_drempel', 'zonder_drempel']:

                    sname = scenario['name']
                    print(f'processing cluster {cluster["cluster"]} for boezem {cluster['boezem']} for scenario {sname}')

                    evenw_peil = cluster[f'evenw_{drempel}_{sname}'] / 100
                    raster = src.read(1)
                    raster_meta = src.meta

                    # grids with data
                    # set nan values to nodata
                    raster = np.where(raster == raster_meta['nodata'], np.nan, evenw_peil - raster)
                    # set all values lower than 0 to nan
                    raster = np.where(raster < 0, np.nan, raster)
                    raster[np.isnan(raster)] = raster_meta['nodata']

                    # export inundation grid
                    output_file = os.path.join(
                        inundation_dir,
                        f'inundation_cluster_{cluster["cluster"]}_boezem_{cluster["boezem"]}_{drempel}_{sname}.tif')

                    if int(cluster['cluster']) in achter_settings:
                        achter_setting = achter_settings[cluster['cluster']]
                        achter_polders_1 = int(achter_setting['achter_polders_1'])
                        achter_polders_2 = int(achter_setting['achter_polders_2'])
                        evenw_peil_achter = cluster[f'evenw_{drempel}_{sname}_achter'] / 100

                        grid_file_achter_1 = os.path.join(dest_dir_clipped, 'cluster_{0}.tif'.format(achter_polders_1))
                        grid_file_achter_2 = os.path.join(dest_dir_clipped, 'cluster_{0}.tif'.format(achter_polders_2))

                        with rasterio.open(os.path.join(tmp_dir, 'raster1.tif'), 'w', **raster_meta) as dst:
                            dst.write(raster, 1)

                        with rasterio.open(grid_file_achter_1, 'r') as src_achter_1:
                            raster_achter_1 = src_achter_1.read(1)
                            raster_meta_achter_1 = src_achter_1.meta
                            raster_achter_1 = np.where(raster_achter_1 == raster_meta_achter_1['nodata'], np.nan, evenw_peil_achter - raster_achter_1)
                            raster_achter_1 = np.where(raster_achter_1 < 0, np.nan, raster_achter_1)
                            raster_achter_1[np.isnan(raster_achter_1)] = raster_meta_achter_1['nodata']
                            with rasterio.open(os.path.join(tmp_dir, 'raster2.tif'), 'w', **raster_meta_achter_1) as dst:
                                dst.write(raster_achter_1, 1)

                        with rasterio.open(grid_file_achter_2, 'r') as src_achter_2:
                            raster_achter_2 = src_achter_2.read(1)
                            raster_meta_achter_2 = src_achter_2.meta
                            raster_achter_2 = np.where(raster_achter_2 == raster_meta_achter_2['nodata'], np.nan, evenw_peil_achter - raster_achter_2)
                            raster_achter_2 = np.where(raster_achter_2 < 0, np.nan, raster_achter_2)
                            raster_achter_2[np.isnan(raster_achter_2)] = raster_meta_achter_2['nodata']
                            with rasterio.open(os.path.join(tmp_dir, 'raster3.tif'), 'w', **raster_meta_achter_2) as dst:
                                dst.write(raster_achter_2, 1)

                        # merge rasters
                        grids = [os.path.join(tmp_dir, 'raster1.tif'), os.path.join(tmp_dir, 'raster2.tif'), os.path.join(tmp_dir, 'raster3.tif')]
                        gdal_merge.main(['', '-o', output_file, '-a_nodata', '-3.40282306e+38'] + grids)
                    else:
                        with rasterio.open(output_file, 'w', **raster_meta) as dst:
                            dst.write(raster, 1)

                    batch.append({
                        'cluster': str(cluster['cluster']),
                        'boezem': str(cluster['boezem']),
                        'comp_keringen': sname,
                        'waterniveau': 'evenw',
                        'peil': evenw_peil,
                        'peil_achter': cluster[f'evenw_{drempel}_{sname}_achter'],
                        'drempel': drempel,
                        'waterniveau_field': f'evenw_{drempel}_{sname}',
                        'inundatie_file': os.path.basename(output_file)
                    })

    # save batch
    df = pd.DataFrame(batch)
    df.to_csv(batch_file, index=False, columns=['cluster', 'cluster_int', 'boezem', 'comp_keringen', 'waterniveau', 'peil', 'drempel', 'waterniveau_field', 'inundatie_file'])
    return
    # make_asc files as input for schadeberekening
    for index, row in df.iterrows():
        tiff_file = row['inundatie_file']
        # make ascii file from tiff file
        ascii_file = os.path.join(inundation_dir_asc, f'{tiff_file}.asc')
        with rasterio.open(os.path.join(inundation_dir, tiff_file)) as src:
            raster = src.read()
            raster_meta = src.meta
            raster_meta['driver'] = 'AAIGrid'
            # no data to -9999
            raster = np.where(raster == raster_meta['nodata'], -9999, raster)
            raster_meta['nodata'] = -9999
            # del raster_meta['count']

            with rasterio.open(ascii_file, 'w', **raster_meta) as dst:
                dst.write(raster)


def check_inundatie_grids():
    """Sommeert het volume van de inundatiegrids en slaat dit op in de overstromingsgebieden ter controle van de
    schadeberekening
    """

    # calculate volumes of inundation grids
    overstromingsgebieden_met_drempel = gpd.read_file(gpkg_file_path, layer=drempelwaarde_layer_name)
    for index, cluster in overstromingsgebieden_met_drempel.iterrows():
        cluster_id = cluster['cluster']
        boezem = cluster['boezem']

        grid = os.path.join(
                        inundation_dir,
                        f'inundation_cluster_{cluster["cluster"]}_boezem_{cluster["boezem"]}_met_drempel_open.tif')

        with rasterio.open(grid) as src:
            # sum of all cells
            raster = src.read(1)
            raster = np.where(raster == src.meta['nodata'], np.nan, raster)
            volume = np.nansum(raster) * abs(src.meta['transform'][0] * src.meta['transform'][4])

        overstromingsgebieden_met_drempel.loc[index, 'grid_volume'] = volume

    overstromingsgebieden_met_drempel.to_file(gpkg_file_path, layer=drempelwaarde_layer_name, driver='GPKG')


def combine_schade_results():
    # pickup batch file
    calculations = pd.read_csv(batch_file)

    # results in excel file to read in pandas are in range A3:B50  (row 3 is the header)
    results = []

    results_cluster = {}

    for index, row in calculations.iterrows():
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
        cluster = row['cluster']
        dft['cluster'] = cluster
        dft['boezem'] = row['boezem']
        dft['waterniveau'] = row['waterniveau']

        if row['cluster'] not in results_cluster:
            results_cluster[cluster] = {
                'cluster': cluster
            }

        if row['waterniveau'] == 'max':
            results_cluster[cluster]['max'] = dft['Totaal'].values[0]
        else:
            results_cluster[cluster][f'c{row['boezem']}'] = dft['Totaal'].values[0]

        # fdt['inundatie_file'] = inundatie_file
        results.append(dft)

    dfb = pd.DataFrame(
        results_cluster.values()
    )
    dfb.to_csv(os.path.join(script_dir, 'results', 'schades_per_oc.csv'))

    # save to csv
    df = pd.concat(results)
    df.to_csv(os.path.join(script_dir, 'results', 'schades_totaal.csv'))

if __name__ == '__main__':
    interpolate_ahn(os.path.join(dest_dir, 'merged_grid.tif'), os.path.join(dest_dir, 'merged_grid_interpolated.tif'))
    merge_aanafvoergebieden_to_cluster()
    make_raster_for_each_cluster()
    make_overstromingscurves_direct_from_grid()
    get_link_overstromingsgebied_boezemgebied()
    calc_evenwichtspeilen()
    get_inundatie_grids()

    #check_inundatie_grids()
    combine_schade_results()


