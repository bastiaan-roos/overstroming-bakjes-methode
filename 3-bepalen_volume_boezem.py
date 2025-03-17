"""
Bepalen van de volume-boezemniveau curves voor elk boezemcompartiment.
output is een csv bestand met de volume-boezemniveau curve voor elk boezemcompartiment voor gesloten en
open compartimenten.
"""

import os
import shutil

import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from osgeo_utils import gdal_merge
from rasterio.mask import mask
from rasterio.features import shapes
import geopandas as gpd

from .paden import *


def make_raster_for_each_boezemcompartiment():
    os.makedirs(dest_dir_clipped_water, exist_ok=True)
    watergebieden = gpd.read_file(gpkg_file_path, layer=watergebieden_layer_name)

    with rasterio.open(merged_grid_path_compressed) as src:
        raster_meta = src.meta

        for index, cluster in watergebieden.iterrows():
            print(f'processing water cluster {cluster['id']}')
            out_filepath = os.path.join(dest_dir_clipped_water, f'cluster_{cluster['id']}.tif')

            if os.path.isfile(out_filepath):
                continue

            geometry = cluster['geometry']
            # clip grids on areas
            try:
                clipped_grid, out_transform = mask(src, [geometry], crop=True, all_touched=True)
            except ValueError as e:
                print(f'Error: {e}')
                continue

            clipped_grid_meta = raster_meta.copy()
            clipped_grid_meta.update({"driver": "GTiff",
                                      "height": clipped_grid.shape[1],
                                      "width": clipped_grid.shape[2],
                                      "transform": out_transform,
                                      "TILED": True,
                                        "BLOCKXSIZE": 256,
                                        "BLOCKYSIZE": 256,
                                        "COMPRESS": "LZW",
                                        "SPARSE_OK": True,
                                      })

            with rasterio.open(out_filepath, 'w', **clipped_grid_meta) as dst:
                dst.write(clipped_grid)


def make_boezemcompartiment_water_polygon_with_extra_attributes(recreate=False):
    watergebieden = gpd.read_file(gpkg_file_path, layer=watergebieden_layer_name)
    water = gpd.read_file(gpkg_file_path, layer=water_layer_name)

    rows = []

    for index, cluster in watergebieden.iterrows():
        # clip and merge water on cluster geometry
        geometry = cluster['geometry']
        # fix geometry
        geometry = geometry.buffer(0)

        id = cluster['id']
        name = cluster['name']
        water_cluster = gpd.clip(water, geometry, keep_geom_type=True)
        water_cluster = water_cluster.dissolve()
        rows.append({
            'id': id,
            'name': name,
            'geometry': water_cluster['geometry'][0],
        })

    watergebieden_opp = gpd.GeoDataFrame(
        rows,
        geometry='geometry',
        crs=watergebieden.crs
    )

    watergebieden_opp['opp'] = watergebieden_opp['geometry'].area

    if 'opp_raster' not in watergebieden_opp.columns:
        watergebieden_opp['max_peil'] = 0.0

        watergebieden_opp['opp_raster'] = 0.0
        watergebieden_opp['opp_intersec'] = 0.0
        watergebieden_opp['opp_buffer'] = 0.0

        watergebieden_opp['opp_raster'] = 0.0
        watergebieden_opp['acurv_0'] = 0.0
        watergebieden_opp['acurv_10'] = 0.0
        watergebieden_opp['acurv_50'] = 0.0
        watergebieden_opp['acurv_90'] = 0.0
        watergebieden_opp['acurv_100'] = 0.0
        watergebieden_opp['perc_acurv'] = 0.0
        watergebieden_opp['opp_oever'] = 0.0
        watergebieden_opp['drempel'] = 0.0

        watergebieden_opp['max_peil'].astype(np.float32)

        watergebieden_opp['opp_raster'].astype(np.float32)
        watergebieden_opp['opp_intersec'].astype(np.float32)
        watergebieden_opp['opp_buffer'].astype(np.float32)

        watergebieden_opp['acurv_0'].astype(np.float32)
        watergebieden_opp['acurv_10'].astype(np.float32)
        watergebieden_opp['acurv_50'].astype(np.float32)
        watergebieden_opp['acurv_90'].astype(np.float32)
        watergebieden_opp['acurv_100'].astype(np.float32)
        watergebieden_opp['perc_acurv'].astype(np.float32)
        watergebieden_opp['opp_oever'].astype(np.float32)
        watergebieden_opp['drempel'].astype(np.float32)

    if not recreate:
        try:
            watergebieden_existing = gpd.read_file(gpkg_file_path, layer=watergebieden_opp_layer_name)
            watergebieden_opp = watergebieden_opp.set_index('id')
            watergebieden_existing = watergebieden_existing.set_index('id')
            cols = ['acurv_0', 'acurv_10', 'acurv_50', 'acurv_90', 'acurv_100', 'perc_acurv', 'opp_oever', 'drempel']
            watergebieden_opp[cols] = watergebieden_existing[cols]
        except Exception as e:
            print(f'Error: {e}')

    watergebieden_opp.to_file(gpkg_file_path, layer=watergebieden_opp_layer_name, driver='GPKG')


def add_info_to_boezemcompartiment_for_each_cluster():
    """ Voegt oppervlak toe aan `watergebieden_opp_layer_name' en maakt csv met percentielen voor de boezemgebieden
    """
    boezemgebieden = gpd.read_file(gpkg_file_path, layer=watergebieden_opp_layer_name)

    percentiles = range(0, 101, 1)
    rows = []
    for index, cluster in boezemgebieden.iterrows():

        output = {
            'id': cluster['id'],
            'name': cluster['name'],
        }
        grid_file = os.path.join(dest_dir_clipped_water, f'cluster_{cluster['id']}.tif')
        print(f'processing water cluster {cluster['id']} with grid file {grid_file}')
        with rasterio.open(grid_file) as src:
            raster = src.read(1)
            raster_meta = src.meta

            # grids with data
            # set nan values to nodata
            raster = np.where(raster == raster_meta['nodata'], np.nan, raster)
            output['nr'] = len(raster[~np.isnan(raster)])
            output['area'] = output['nr'] * abs(raster_meta['transform'][0] * raster_meta['transform'][4])

            percentiles_result = np.nanpercentile(raster, percentiles)

            for i, percentile in enumerate(percentiles):
                output[percentile] = percentiles_result[i]

            rows.append(output)

            # make polygon of raster
            raster = np.where(np.isnan(raster), 0, 1).astype(np.uint8)

            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(
                shapes(raster,
                       mask=raster,
                       transform=src.transform))
            )

            # make geopandas dataframe
            grid_poly = gpd.GeoDataFrame.from_features(list(results), crs=src.crs)

            # make multipolygon
            multipolygon = grid_poly.union_all()
            multipolygon = multipolygon.buffer(0)

            # get only the buffer of 10 meter, minus the original area
            buffer = 10
            multipolygon_buffer = multipolygon.buffer(buffer).difference(multipolygon)

            out = {
                'geom': multipolygon,
                'buffer': multipolygon_buffer,
                'id': cluster['id'],
                'name': cluster['name'],
            }

            # clip multipolygon to cluster geometry
            geometry = cluster['geometry']
            multipolygon = multipolygon.intersection(geometry)
            multipolygon_buffer = multipolygon_buffer.intersection(geometry)

            # set cluster back to boezemgebieden
            boezemgebieden.loc[index, 'opp_intersec'] = multipolygon.area
            boezemgebieden.loc[index, 'opp_buffer'] = multipolygon_buffer.area
            boezemgebieden.loc[index, 'opp_raster'] = output['area']

            out['geom_clipped'] = multipolygon
            out['buffer_clipped'] = multipolygon_buffer

    df = pd.DataFrame(rows)
    df.to_csv(boezem_curve, index=False)

    boezemgebieden.to_file(gpkg_file_path, layer=watergebieden_opp_layer_name, driver='GPKG')

    # output polygonen van tussenresultaten voor controle
    # gpd.GeoDataFrame([{'id': feat['id'], 'name': feat['name'], 'geometry': feat['geom']} for feat in areas], crs=boezemgebieden.crs).to_file(
    #     gpkg_file_path, layer='check_grid_poly', driver='GPKG')
    # gpd.GeoDataFrame([{'id': feat['id'], 'name': feat['name'], 'geometry': feat['buffer']} for feat in areas], crs=boezemgebieden.crs).to_file(
    #     gpkg_file_path, layer='check_grid_buffer', driver='GPKG')
    # gpd.GeoDataFrame([{'id': feat['id'], 'name': feat['name'], 'geometry': feat['geom_clipped']} for feat in areas], crs=boezemgebieden.crs).to_file(
    #     gpkg_file_path, layer='check_grid_poly_clipped', driver='GPKG')
    # gpd.GeoDataFrame([{'id': feat['id'], 'name': feat['name'], 'geometry': feat['buffer_clipped']} for feat in areas], crs=boezemgebieden.crs).to_file(
    #     gpkg_file_path, layer='check_grid_buffer_clipped', driver='GPKG')


def make_storage_depth_curves_boezem():
    boezemgebieden = gpd.read_file(gpkg_file_path, layer=watergebieden_opp_layer_name)
    boezemcurve = pd.read_csv(boezem_curve)
    boezemcurve.set_index('id', inplace=True)

    for index, cluster in boezemgebieden.iterrows():

        max_peil = int(cluster['max_peil'] * 100)
        opp = cluster['opp']
        opp_raster = cluster['opp_intersec']
        opp_oever = cluster['opp_oever']
        percentage_with_curve = cluster['perc_acurv']

        perc_oever = (opp_oever / opp) * 100

        missing_area = opp - opp_raster - opp_oever
        opp_with_curve = missing_area * percentage_with_curve / 100

        opp_extrapolate = missing_area * (100 - percentage_with_curve) / 100
        factor_extrapolate = 1 + opp_extrapolate / opp_raster

        meas_curve = []

        # make list elevation - area for raster
        for i in range(0, 101, 1):
            field = f'{i}'
            meas_curve.append({
                'elevation': boezemcurve.loc[cluster['id'], field],
                'area': float(i) * opp_raster * factor_extrapolate / 100
            })

        # make list elevation - area for additional curve
        additional_curve = [
            {'elevation': cluster['acurv_0'], 'add_area': 0},
            {'elevation': cluster['acurv_10'], 'add_area': opp_with_curve * 0.10},
            {'elevation': cluster['acurv_50'], 'add_area': opp_with_curve * 0.50},
            {'elevation': cluster['acurv_90'], 'add_area': opp_with_curve * 0.90},
            {'elevation': cluster['acurv_100'], 'add_area': opp_with_curve},
        ]

        oever_curve = [
            {'elevation': -125, 'oever_area': 0},
            {'elevation': -75, 'oever_area': opp_oever},
        ]

        min_elevation = min(meas_curve[0]['elevation'], additional_curve[0]['elevation'], -5.0)
        max_elevation = max(meas_curve[-1]['elevation'], additional_curve[-1]['elevation'], max_peil/ 100)
        min_elevation = int(min_elevation * 100)
        max_elevation = int(max_elevation * 100)

        min_elevation = max(min_elevation, -1000)
        max_elevation = min(max_elevation, max_peil)

        df_meas_curve = pd.DataFrame(
            meas_curve,
        )
        # make elevation to cm and round
        df_meas_curve['elevation'] = df_meas_curve['elevation'] * 100
        df_meas_curve['elevation'] = df_meas_curve['elevation'].round(0).astype(int)
        # set min to -1000 and max to 0
        df_meas_curve['elevation'] = df_meas_curve['elevation'].clip(lower=-1000, upper=0)

        # remove double elevation levels, keep first (largest)
        df_meas_curve = df_meas_curve.drop_duplicates(subset='elevation', keep='first')
        df_meas_curve.set_index('elevation', inplace=True)

        df_tot_meas_curve = pd.DataFrame(
            [{'elevation': el, 'area': np.nan} for el in range(max_elevation, min_elevation - 1, -1)]
        )
        df_tot_meas_curve.set_index('elevation', inplace=True)
        df_tot_meas_curve.update(df_meas_curve)
        del df_meas_curve
        df_tot_meas_curve = df_tot_meas_curve.interpolate(method='index')
        df_tot_meas_curve = df_tot_meas_curve.bfill()
        df_tot_meas_curve['volume_cm'] = df_tot_meas_curve['area'] * 0.01
        # make cumulative
        df_tot_meas_curve['volume'] = df_tot_meas_curve['volume_cm'].cumsum()

        df_add_curve = pd.DataFrame(
            additional_curve
        )
        df_add_curve['elevation'] = df_add_curve['elevation'] * 100
        df_add_curve['elevation'] = df_add_curve['elevation'].round(0).astype(int)
        # set min to -1000 and max to 0
        df_add_curve['elevation'] = df_add_curve['elevation'].clip(lower=-1000, upper=max_peil)

        df_add_curve = df_add_curve.drop_duplicates(subset='elevation', keep='last')
        df_add_curve.set_index('elevation', inplace=True)

        df_tot_add_curve = pd.DataFrame(
            [{'elevation': el, 'add_area': np.nan} for el in range(max_elevation, min_elevation - 1, -1)]
        )
        df_tot_add_curve.set_index('elevation', inplace=True)
        df_tot_add_curve.update(df_add_curve)
        del df_add_curve
        df_tot_add_curve = df_tot_add_curve.interpolate(method='index')
        # also fill the nan rows before with the first value
        df_tot_add_curve = df_tot_add_curve.bfill()

        df_tot_add_curve['add_volume_cm'] = df_tot_add_curve['add_area'] * 0.01
        # make cumulative
        df_tot_add_curve['add_volume'] = df_tot_add_curve['add_volume_cm'].cumsum()

        df_add_oever = pd.DataFrame(
            oever_curve
        )
        df_add_oever.set_index('elevation', inplace=True)
        df_tot_add_oever = pd.DataFrame(
            [{'elevation': el, 'oever_area': np.nan} for el in range(max_elevation, min_elevation - 1, -1)]
        )
        df_tot_add_oever.set_index('elevation', inplace=True)
        df_tot_add_oever.update(df_add_oever)
        del df_add_oever
        df_tot_add_oever = df_tot_add_oever.interpolate(method='index')
        # also fill the nan rows before with the first value
        df_tot_add_oever = df_tot_add_oever.bfill()
        df_tot_add_oever['oever_volume_cm'] = df_tot_add_oever['oever_area'] * 0.01
        # make cumulative
        df_tot_add_oever['oever_volume'] = df_tot_add_oever['oever_volume_cm'].cumsum()

        df_curve = pd.concat([df_tot_meas_curve, df_tot_add_curve, df_tot_add_oever], axis=1)
        df_curve['volume_tot'] = df_curve['volume'].fillna(0) + df_curve['add_volume'].fillna(0) + df_curve['oever_volume'].fillna(0)
        df_curve['area_tot'] = df_curve['area'].fillna(0) + df_curve['add_area'].fillna(0) + df_curve['oever_area'].fillna(0)

        # remove all below -5.0 and above max_peil
        df_curve = df_curve[(df_curve.index >= -500) & (df_curve.index <= max_peil)]
        df_curve.to_csv(os.path.join(output_path_boezem, f'total_curve_{cluster["id"]}.csv'), index=True)

        # find first elevation where area_tot > 10% of opp for drempel
        try:
            boezemgebieden.loc[index, 'drempel'] = df_curve[df_curve['area_tot'] < 0.1 * opp_raster].index[0]
        except IndexError:
            boezemgebieden.loc[index, 'drempel'] = -500

    boezemgebieden.to_file(gpkg_file_path, layer=watergebieden_opp_layer_name, driver='GPKG')


def make_curves_open():
    boezemgebieden = gpd.read_file(gpkg_file_path, layer=watergebieden_opp_layer_name)

    clusters = []
    total_curve = None
    sum_fields = []

    for index, cluster in boezemgebieden.iterrows():
        cluster_id = cluster['id']
        clusters.append(cluster_id)
        # get_curve
        df_curve = pd.read_csv(os.path.join(output_path_boezem, f'total_curve_{cluster_id}.csv'))
        df_curve.set_index('elevation', inplace=True)
        df_curve = df_curve[['volume_tot']]
        df_curve.rename(columns={'volume_tot': f'vol_{cluster_id}'}, inplace=True)

        if total_curve is None:
            total_curve = df_curve
        else:
            total_curve = total_curve.merge(df_curve, on='elevation', how='outer')

        sum_fields.append(f'vol_{cluster_id}')

    # fill nans with 0
    total_curve.fillna(0, inplace=True)
    total_curve.sort_index(ascending=False, inplace=True)

    drempels = {}

    # make curves when open
    for index, cluster in boezemgebieden.iterrows():
        cluster_id = cluster['id']
        drempels[cluster_id] = cluster['drempel']


    for cluster_id in clusters:
        tot_open_curve = pd.DataFrame(total_curve.reset_index()['elevation'].copy())
        tot_open_curve.set_index('elevation', inplace=True)
        tot_open_curve[f'volume_tot'] = 0

        for linked_cluster_id in clusters:
            if linked_cluster_id == cluster_id:
                drempel = -501
            else:
                drempel = max(drempels[cluster_id], drempels[linked_cluster_id])

            try:
                volume_on_drempel = total_curve.loc[drempel, f'vol_{linked_cluster_id}']
            except KeyError:
                volume_on_drempel = total_curve.loc[-500, f'vol_{linked_cluster_id}']

            tot_open_curve[f'volume_tot'] += (total_curve[f'vol_{linked_cluster_id}']
                                            .where(total_curve[f'vol_{linked_cluster_id}'].index > drempel, volume_on_drempel))

        # calc diff for each cm
        tot_open_curve['volume_delta'] = tot_open_curve['volume_tot'] - tot_open_curve['volume_tot'].shift(1)

        total_curve[f'vol_open={cluster_id}'] = tot_open_curve['volume_tot']

        tot_open_curve.to_csv(os.path.join(output_path_boezem, f'total_curve_open_{cluster_id}.csv'), index=True)

    # combine curves of all boezems for review
    total_curve.to_csv(os.path.join(output_path_boezem, 'total_curves.csv'), index=True)


if __name__ == '__main__':
    make_raster_for_each_boezemcompartiment()
    make_boezemcompartiment_water_polygon_with_extra_attributes()
    # --> vul de perc_acurv en de acurv in in de boezemgebieden
    # add_info_to_boezemcompartiment_for_each_cluster()
    make_storage_depth_curves_boezem()
    # --> vul 'dempel' in in de boezemgebieden
    make_curves_open()
