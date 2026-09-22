#!/bin/bash
#SBATCH -J BP5-30km
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --export=ALL
#SBATCH -o BP5.o%j
#SBATCH -e BP5.e%j
# Supply your allocation with sbatch -A ACCOUNT. Submit from the copied job directory.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the job directory}"

# Load the EXACT compiler/MPI/deal.II modules used for this server build
# before submission (or insert those known commands here). Do not substitute
# a guessed module stack. The packaged local .so is NOT a portable server build.
module list
export SLURM_EXPORT_ENV=ALL
source ./environment.sh
BP5_EXECUTABLE=${BP5_EXECUTABLE:-./aspect-release}
BP5_PARAMETERS=${BP5_PARAMETERS:-first_event.prm}
python3 verify_package.py
test -x "$BP5_EXECUTABLE"
test -r ./libbp5_initialization.release.so
test -r "$BP5_PARAMETERS"
sha256sum "$BP5_EXECUTABLE" ./libbp5_initialization.release.so "$BP5_PARAMETERS"
ibrun "$BP5_EXECUTABLE" "$BP5_PARAMETERS"
