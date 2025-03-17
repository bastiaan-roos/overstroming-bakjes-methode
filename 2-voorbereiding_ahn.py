"""
script voor het downloaden van AHN tiles van pdok deze te combineren tot één geotiff
"""

import httpx
from httpx import ConnectTimeout, ReadTimeout
from osgeo import ogr, gdal
from osgeo_utils import gdal_merge

from .paden import *

ogr.UseExceptions()


def download_grids():
    # download script
    ds = ogr.Open(kaartbladen_file, 0)
    layer = ds.GetLayer(kaartbladen_layer_name)
    layer.ResetReading()

    for feature in layer:
        if not feature[kaartbladen_field_do_download]:
            continue

        link = feature[field_download_link]
        if link is None:
            print("Error: no link found for feature {0}".format(feature.GetFID()))
            continue
        file_name = os.path.basename(link)
        existing_files = [f.lower() for f in os.listdir(dest_dir)]
        if file_name.lower() not in existing_files and file_name.lower().replace('.zip', '.tif') not in existing_files:
            # download file
            print('downloading {0}'.format(link))
            try:
                with httpx.stream('GET', link, verify=False, timeout=120) as r:
                    with open(os.path.join(dest_dir, file_name), 'wb') as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)

                print('ready downloading {0}'.format(file_name))
            except (ConnectTimeout, ReadTimeout) as e:
                print('Error downloading {0}: {1}'.format(link, e))

        else:
            print('{0} already present'.format(file_name))

    print('ready')


def combine_downloaded_grids_with_vtr():

    tiff_files = []
    # get tiff files
    for f in os.listdir(dest_dir):
        if f.lower().endswith('.tif'):
            tiff_files.append(os.path.join(dest_dir, f))
    vrt_file = os.path.join(dest_dir, 'ahn.vrt')
    gdal.BuildVRT(vrt_file, tiff_files)  #, options=gdal.BuildVRTOptions(separate=True))

    print('ready')


def combine_grids_to_one_geotiff():

    merged_grid = os.path.abspath(os.path.join(dest_dir, 'merged_grid.tif'))
    # remove merged grid if exists
    if os.path.isfile(merged_grid):
        os.remove(merged_grid)
    grids = [os.path.abspath(os.path.join(dest_dir, f)) for f in os.listdir(dest_dir) if f.lower().endswith('.tif')]
    gdal_merge.main(['', '-o', merged_grid, '-a_nodata', '-3.40282306e+38'] + grids)
