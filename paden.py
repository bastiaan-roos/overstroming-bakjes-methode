import os

script_dir = os.path.dirname(__file__)
tmp_dir = os.path.join(script_dir, 'tmp')

### geopackage met lagen
gpkg_file_path = os.path.join(script_dir, 'data', 'gebieden.gpkg')

# voor overstromingsgebieden
aanafvoergebieden_layer_name = 'aanafvoergebieden'
aanafvoergebieden_clustering_column = 'cluster'
overstromingsgebieden_layer_name = 'overstromingsgebieden'

# voor boezemcompartimenten
watergebieden_layer_name = 'water_gebieden'
water_layer_name = 'water'
watergebieden_opp_layer_name = 'watergebieden_opp'

# gecombineerd
drempelwaarde_layer_name = 'drempelwaarden'

### overstromingsgebied
# voorbereiding
# geopackage met ahn kaartbladen en welke te downloaden
kaartbladen_file = os.path.join(script_dir, 'data', 'kaartbladen.gpkg')
kaartbladen_layer_name = 'EllipsisDrive_index_fancy'
kaartbladen_field_do_download = 'download'
do_5m = True
if do_5m:
    # 5m
    field_download_link = 'AHN5 maaiveldmodel (DTM) 5m'
    dest_dir = os.path.join(script_dir, 'data', 'ahn_5m')
else:
    # 0.5m
    field_download_link = 'AHN5 maaiveldmodel (DTM) ½m'
    dest_dir = os.path.join(script_dir, 'data', 'ahn')

merged_grid_file = os.path.join(dest_dir, 'merged_grid.tif')

# bewerkingen vanaf gecombineerd grid
# grid met interpolatie
merged_interpolated_grid_file = os.path.join(dest_dir, 'merged_grid_interpolated.tif')
# grids per overstromingsgebied
dest_dir_clipped = os.path.join(dest_dir, 'ahn_clipped')
# maaiveldcurve percentielen
maaiveldcurve_filepath = os.path.join(dest_dir, 'maaiveldcurve.csv')

# waterdiepte grids (tif) per overstromingsgebied en boezemcompartiment
inundation_dir = os.path.join(dest_dir, 'inundation')
# waterdiepte grids (asc) voor input schadeberekening
inundation_dir_asc = os.path.join(dest_dir, 'inundation_asc')


### boezemcompartimenten
# pad met de originele aangeleverde data
input_path = os.path.join(script_dir, 'data', 'water_grid', 'input')
# pad met aangeleverde data die gecorrigeerd is in een eenduidig xyz formaat dat kan worden ingelezen door GDAL
preprocessed_path = os.path.join(script_dir, 'data', 'water_grid', 'preprocessed')
# pad waar de uiteindelijke csv's mer curves inkomen
output_path_boezem = os.path.join(script_dir, 'data', 'water_grid')
# gecombineerd grid van de boezem
merged_grid_path_compressed = os.path.abspath(os.path.join(script_dir, 'data', 'water_merged_grid_compressed.tif'))
# grids per boezemcompartiment
dest_dir_clipped_water = os.path.abspath(os.path.join(output_path_boezem, 'water_clipped'))
# percentielen boezemgebieden
boezem_curve = os.path.join(output_path_boezem, 'boezemcurve.csv')

# gecombineerd grid van de boezem
batch_file = os.path.join(script_dir, 'data', 'batch.csv')

# resultaten SSM
result_folder = os.path.join(script_dir, 'results')

# Create output directory if it does not exist
os.makedirs(preprocessed_path, exist_ok=True)
os.makedirs(output_path_boezem, exist_ok=True)

os.makedirs(dest_dir, exist_ok=True)
os.makedirs(dest_dir_clipped, exist_ok=True)
os.makedirs(inundation_dir, exist_ok=True)
os.makedirs(inundation_dir_asc, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)
