#!/bin/bash
#PBS -N miniqg_training
#PBS -l walltime=12:00:00
#PBS -l select=8:ncpus=64:ngpus=4:mem=235GB
#PBS -j oe
#PBS -o miniqg_training.out

PYTHON_PATH=/path/to/your/conda-env/bin/python

export TMPDIR=/path/to/scratch/tmpdir
mkdir -p $TMPDIR

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=WARN

NUM_NODES=$(cat $PBS_NODEFILE | sort -u | wc -l)
GPUS_PER_NODE=4
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500
export WORLD_SIZE=$TOTAL_GPUS

echo "======================================"
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Nodes: $NUM_NODES  |  Total GPUs: $TOTAL_GPUS"
echo "======================================"

cd /path/to/your/project/

MPIEXEC=$(which mpiexec 2>/dev/null || echo "/opt/cray/pe/pals/1.2.12/bin/mpiexec")

CONDA_LIB=/path/to/your/conda-env/lib

$MPIEXEC --envall -n $TOTAL_GPUS -ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE \
    bash -c "
    export LD_LIBRARY_PATH=$CONDA_LIB:\$LD_LIBRARY_PATH

    export RANK=\$PALS_RANKID
    export LOCAL_RANK=\$PALS_LOCAL_RANKID
    export WORLD_SIZE=$TOTAL_GPUS
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=29500

    export CUDA_VISIBLE_DEVICES=0,1,2,3

    $PYTHON_PATH training/train.py
    "

EXIT_CODE=$?

echo "======================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

exit $EXIT_CODE