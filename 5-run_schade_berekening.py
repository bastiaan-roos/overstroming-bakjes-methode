"""
Schade berekening voor alle scenario's beschreven in batch_file.
Voer deze onder windows uit
Zet de his ssm in `/Fiat_int` map


"""


import os
import sys
import shutil

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from paden import script_dir, batch_file, inundation_dir_asc, result_folder


input_folder = os.path.join(script_dir, 'input')
output_folder = os.path.join(script_dir, 'output')

os.makedirs(input_folder, exist_ok=True)



def run_schade_berekening():

    calculations = pd.read_csv(batch_file)

    for index, row in calculations.iterrows():

        # clean input folder
        for file in os.listdir(input_folder):
            os.remove(os.path.join(input_folder, file))

        ascii_file = os.path.join(inundation_dir_asc, row['inundatie_file'] + '.asc')

        out_path = os.path.join(result_folder, row['inundatie_file'][:-4])
        if os.path.exists(os.path.join(out_path, "schades.txt")):
            print(f"Skipping {index}")
            continue

        # run schadeberekening
        hissssm_exe = os.path.join(script_dir, 'Fiat_int', 'x64', 'Delft-Fiat.exe')

        shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        # set paths in commandline, so excel could be static
        # fixed 5x5 selection

        command = (f'{hissssm_exe} {os.path.join(script_dir, 'start_FIAT_agv.xlsx')}  '
                   f'--result_name schades '
                   f'--hazard_filepath "{ascii_file}" '
                   f'--functions_directory "{os.path.join(script_dir, "Fiat_int", "2023", "functies")}" '
                   f'--exposure_directory {os.path.join(script_dir, "Fiat_int", "2023")} '
                   f'--output_directory "{output_folder}" '
                   f'--log_filepath "{os.path.join(output_folder, "log.txt")}" '
                   )

        # print(command)
        os.system(f'start /wait cmd /c {command} ')

        # move results
        os.makedirs(out_path, exist_ok=True)
        files = [
            "log.txt",
            "schades.txt",
            "schades.xls",
            "Totaalschade.tif"
        ]
        for file in files:
            out_file_path = os.path.join(out_path, file)
            if os.path.exists(out_file_path):
                os.remove(out_file_path)
            shutil.move(os.path.join(output_folder, file), out_path)


if __name__ == '__main__':
    run_schade_berekening()
