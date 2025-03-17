"""
Functies om aangeleverde data om te zetten naar een uniform formaat dat kan worden ingelezen door GDAL.
Eindproduct is een gecombineerde geotiff van alle grids (0.25m resolutie).
"""
from osgeo import gdal
from osgeo_utils import gdal_merge
import pandas as pd
import rasterio

from .paden import *


def remove_dot_thousand_seperator_and_change_comma_to_dot_and_make_space_seperated(line):
    # 130078.800,   476141.900,       -0.730  ==> 130078.800 476141.900 -0.730
    # 132922,5	478667,5	-1,69  ==> 132922.5 478667.5 -1.69
    # 111.605,50	449.877,50	-3,15  ==> 111.60550 449.87750 -3.15
    # 132922.5 478667.5 -1.69  ==> 132922.5 478667.5 -1.69
    # 130078.800,   476141.900,       -0.730  ==> 130078.800 476141.900 -0.730
    # 128719.500,   465229.500,       -1.720  ==> 128719.500 465229.500 -1.720
    # 132922.5 478667.5 -1.69  ==> 132922.5 478667.5 -1.69
    # 132922,5	478667,5	-1,69  ==> 132922.5 478667.5 -1.69
    line = line.replace('.', '')
    return change_comma_to_dot_and_make_space_seperated(line)


def change_comma_to_dot_and_make_space_seperated(line):
    line = line.replace(',', '.')
    return make_space_seperated(line)


def make_comma_space_seperated(line):
    line = line.replace(',', ' ')
    # greedy remove spaces
    return make_space_seperated(line)


def make_space_seperated(line):
    line = line.replace('\t', ' ')
    # greedy remove spaces
    while '  ' in line:
        line = line.replace('  ', ' ')
    if line[0] == ' ':
        line = line[1:]
    return line


def special_case(line):
    values = line.split('\t')
    if len(values) < 3:
        return ''
    values = [value.strip() for value in values if value.strip()]
    values[0] = values[0].replace('.', '')
    values[1] = values[1].replace('.', '')
    values[2] = values[2].replace(',', '.')
    line = ' '.join(values) + '\n'
    return line


def get_change_function(line):
    # several examples:
    # 132922,5	478667,5	-1,69  ==> comma decimal
    # 132922.5 478667.5 -1.69  ==> dot decimal
    # 111.605,50	449.877,50	-3,15  ==> comma decimal
    # 130078.800,   476141.900,       -0.730  ==> dot decimal
    # return True if dot decimal, False if comma decimal
    nr_of_commas = line.count(',')
    nr_of_dots = line.count('.')
    comma_space_seperated = line.count(', ')
    comma_tab_seperated = line.count(',\t')
    nr_of_spaces = line.count(' ')
    nr_of_tabs = line.count('\t')
    first_place_of_comma = line.find(',')
    first_place_of_dot = line.find('.')
    if nr_of_commas == 0 and nr_of_dots == 0:
        return make_space_seperated
    if nr_of_commas == 0 and nr_of_dots > 0:
        return make_space_seperated
    if comma_space_seperated or comma_tab_seperated or (nr_of_spaces == 0 and nr_of_tabs == 0):
        return make_comma_space_seperated
    if nr_of_commas > 0 and nr_of_dots == 0:
        return change_comma_to_dot_and_make_space_seperated
    if nr_of_commas > 0 and nr_of_dots > 0:
        if first_place_of_dot < first_place_of_comma:
            return remove_dot_thousand_seperator_and_change_comma_to_dot_and_make_space_seperated
        else:
            return change_comma_to_dot_and_make_space_seperated


def process_xyz_file(input_file_path, output_file_path):
    # check header
    with open(input_file_path, 'r') as f:
        with open(output_file_path, 'w') as f_out:
            first_line = f.readline()
            second_line = f.readline()
            third_line = f.readline()

            number_line = second_line

            # first add header
            f_out.write('x y z\n')
            if '===' in third_line.lower():
                for _ in range(6):
                    number_line = f.readline()
                f.seek(0)
                for _ in range(5):
                    f.readline()
            elif 'x' in first_line.lower():
                f.readline()

            if os.path.basename(input_file_path) == 'P3060_160415_1x1_Singelgracht saneringslocatie gasfabriek_.txt':
                change_function = special_case
            else:
                change_function = get_change_function(number_line)

            for line in f:
                if line[0].lower() == 'e':
                    continue
                f_out.write(change_function(line))

    # reopen as csv (space seperated) with pandas and sort by y and x
    df = pd.read_csv(output_file_path, sep=' ')
    # remove duplicates in x and y descending
    df = df.sort_values(by=['y', 'x'], ascending=[False, True])

    # remove lines wher z is empty
    df = df[df['z'].notna()]
    # remove duplicates in x and y
    df = df.drop_duplicates(subset=['x', 'y'], keep='last')
    df.to_csv(output_file_path, sep=' ', index=False)


def prepare_waterdepth_grid():
    # loop over all files in the input directory
    # if 'asc' file, copy to preprocessed directory
    # if txt, based on first line
    #   if 'xllcorner', copy to preprocessed directory and change extension to asc
    #   if header like:
    """
Layer : <something>
<something>
========================================
   Easting       Northing        Height
========================================
    """
    # replace with Header: x,y,z
    #   if first line include x y and z (caseinsensitive) or directly only numbers, write as xyz extension and change content:
    #       if second line like 132922,5	478667,5	-1,69  => make space seperated and change comma to dot
    #       if second line like 132922.5 478667.5 -1.69  => make space seperated and change comma to dot
    #       if second line like 111.605,50	449.877,50	-3,15  => make space seperated and remove dot and change comma to dot
    #       if second line like  130078.800,   476141.900,       -0.730  => make space seperated
    #       if second line like    128719.500,   465229.500,       -1.720  => make space seperated
    #       if second line like  132922.5 478667.5 -1.69  => make space seperated and change comma to dot
    # if .asc and firstline not xllcorner, change extension to xyz and copy to preprocessed directory

    filenames = os.listdir(input_path)
    filenames.sort()

    for filename in filenames:
        print(filename)
        with open(os.path.join(input_path, filename), 'r') as f:
            first_line = f.readline().lower()

        if filename.endswith('.asc'):
            if 'ncols' in first_line:
                pass
                # shutil.copyfile(os.path.join(input_path, filename), os.path.join(preprocessed_path, filename))
            else:
                pass
                process_xyz_file(
                    os.path.join(input_path, filename),
                    os.path.join(preprocessed_path, filename.replace('.asc', '.xyz'))
                )

        elif filename.endswith('.txt'):
            if 'ncols' in first_line:
                pass
                # shutil.copyfile(os.path.join(input_path, filename), os.path.join(preprocessed_path, filename.replace('.txt', '.asc')))
            else:
                pass
                process_xyz_file(
                    os.path.join(input_path, filename),
                    os.path.join(preprocessed_path, filename.replace('.txt', '.xyz'))
                )


def make_geotiff_from_xyz_and_asc_files():
    filenames = os.listdir(preprocessed_path)
    filenames.sort()
    # filenames.reverse()
    for filename in filenames:
        try:
            print(filename)

            translate_options = gdal.TranslateOptions(
                format="GTiff",
                outputSRS="EPSG:28992",  # Of targetSRS="EPSG:28992"
                creationOptions=["TILED=YES", "COMPRESS=LZW", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "SPARSE_OK=TRUE"]
            )

            gdal.Translate(
                os.path.join(output_path_boezem, filename[:-3] + 'tif'),
                os.path.join(preprocessed_path, filename),
                options=translate_options
            )

        except Exception as e:
            print(e)


def fix_rasters_upside_down():
    # input files
    input_files = [os.path.join(output_path_boezem, f) for f in os.listdir(output_path_boezem) if f.endswith('.tif')]
    input_files.sort()

    for input_file in input_files:
        with rasterio.open(input_file) as src:
            data = src.read()
            transform = src.transform

            if transform.e < 0:
                continue

            print(f'Fixing {input_file}')
            # Keer de data om (alle banden)
            flipped_data = data[:, ::-1, :] if data.ndim > 2 else data[:, ::-1]

            # Bereken de nieuwe oorsprong (x, y)
            new_origin_x = transform.c
            new_origin_y = transform.f + src.height * transform.e  # Aangepast!

            # Pas de transform aan
            flipped_transform = rasterio.Affine(
                transform.a, transform.b, new_origin_x,
                transform.d, -transform.e, new_origin_y
            )

            # Schrijf het omgedraaide bestand
            with rasterio.open(input_file, 'w', driver='GTiff', height=src.height,
                               width=src.width, count=src.count, dtype=data.dtype, crs=src.crs,
                               transform=flipped_transform, nodata=src.nodata) as dst:
                dst.write(flipped_data)


def combine_grids_to_one_geotiff():
    # tijdelijk bestand voor het samenvoegen van de grids (ongecomprimeerd, is nogal groot)
    merged_grid_path = os.path.abspath(os.path.join(script_dir, 'data', 'water_merged_grid.tif'))

    # remove merged grid if exists
    if os.path.isfile(merged_grid_path):
        os.remove(merged_grid_path)

    gdal.UseExceptions()

    grids = [f'{os.path.abspath(os.path.join(output_path_boezem, f))}' for f in os.listdir(output_path_boezem) if f.lower().endswith('.tif')]
    grids.sort()
    gdal_merge.main(['',
                     '-o', merged_grid_path,
                     '-a_nodata', '-3.40282306e+38',
                     '-ps', '0.25', '-0.25'] + grids)

    gdal.Translate(
        merged_grid_path_compressed,
        merged_grid_path,
        format='GTiff',
        creationOptions=[
            'TILED=YES',
            'COMPRESS=LZW',
            'BLOCKXSIZE=256',
            'BLOCKYSIZE=256',
            'SPARSE_OK=TRUE'
        ]
    )

    os.remove(merged_grid_path)


if __name__ == '__main__':
    prepare_waterdepth_grid()
    make_geotiff_from_xyz_and_asc_files()
    fix_rasters_upside_down()
    combine_grids_to_one_geotiff()
