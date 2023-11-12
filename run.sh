#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH -o outputs-%j

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/mp3Env/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate mp3Env

# wandb disabled 
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"
export WANDB_CACHE_DIR="/scratch/general/vast/u1419542/wandb_cache"
export CUDA_LAUNCH_BLOCKING=1

SIZE="tiny"
DATASET="sst2"
EVAL_ONLY=false
TEST_ONLY=false
TEST_FILE="/uufs/chpc.utah.edu/common/home/u1419542/CS6957/mp4/data/hidden_sst2.csv"
LOAD_MODEL="/uufs/chpc.utah.edu/common/home/u1419542/CS6957/mp4/models/BERT_mini_rte.pt"
while getopts 'd:es:tf:l:' opt; do
  case "$opt" in
    d)   DATASET="$OPTARG"     ;;
    e)   EVAL_ONLY=true     ;;
    s)  SIZE="$OPTARG"   ;;
    t)   TEST_ONLY=true     ;;
    f)   TEST_FILE="$OPTARG" ;;
    l)  LOAD_MODEL="$OPTARG" ;;
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

if [ "$TEST_ONLY" = true ] ; then 
    python3 model.py -info -testOnly -size $SIZE -dataset $DATASET -test $TEST_FILE -load $LOAD_MODEL
else
  if [ "$EVAL_ONLY" = true ] ; then 
    python3 model.py -info -size $SIZE -dataset $DATASET -skipTraining
  else
    python3 model.py -info -size $SIZE -dataset $DATASET
  fi  ;
fi  ;