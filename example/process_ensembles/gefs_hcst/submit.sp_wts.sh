#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/zhennanshi/logs/sp_wts.out
#SBATCH --error=/scratch/zhennanshi/logs/sp_wts.err
#SBATCH --account=2306081036
#SBATCH --job-name=sp_wts

####zhennan rvised version
cd /u/au/ac/zhennanshi/znprojects/generic_deltamodel/example/process_ensembles/gefs_hcst || exit 1

mkdir -p /scratch/zhennanshi/logs

eval "$(conda shell.bash hook)"
conda activate dmg312

python poly2poly_py3.py \
  /beegfs/sets/aw-ciroh/common/gis/camels_revised_awood/gagesII_671_shp_albers.clean.gpkg \
  GAGE_ID \
  /beegfs/sets/aw-ciroh/common/gis/grids/GEFS_v12.25d.xconus.gpkg \
  id \
  GRID \
  ./mapping_gefs_xconus_to_camels_gII.nc