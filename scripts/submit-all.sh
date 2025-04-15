#!/bin/bash
# Hard-coded checkpoint and cluster values
NEW_CHECKPOINT="jaslee20/llava-epoch3-qwen2-unified"
CLUSTER_ARG="saturn"

# Determine the new cluster constraint based on the hard-coded input.
if [ "$CLUSTER_ARG" == "jupiter" ]; then
  NEW_CLUSTER="ai2/jupiter-cirrascale-2"
elif [ "$CLUSTER_ARG" == "neptune" ]; then
  NEW_CLUSTER="ai2/neptune-cirrascale"
elif [ "$CLUSTER_ARG" == "saturn" ]; then
  NEW_CLUSTER="ai2/saturn-cirrascale"
elif [ "$CLUSTER_ARG" == "ceres" ]; then
  NEW_CLUSTER="ai2/ceres-cirrascale"
else
  echo "Unsupported cluster: $CLUSTER_ARG"
  exit 1
fi

echo "Using new checkpoint: ${NEW_CHECKPOINT}"
echo "Using new cluster constraint: ${NEW_CLUSTER}"

# List of experiment directories.
EXPERIMENT_DIRS=("/data/input/jiafei/GroundedVLA/SimplerEnv/scripts/open_drawer")

# Loop through each experiment directory.
for dir in "${EXPERIMENT_DIRS[@]}"; do
  echo "Processing directory: $dir"
  
  # --- Update checkpoint paths in bash scripts ---
  for script in "$dir"/*.sh; do
    if [ -f "$script" ]; then
      # Replace for lines starting with "declare -a arr="
      sed -i.bak -E "s@^(declare -a arr=\(\")[^\"]+(\")@\1${NEW_CHECKPOINT}\2@g" "$script" || true
      # Replace for lines starting with "declare -a ckpt_paths="
      sed -i.bak -E "s@^(declare -a ckpt_paths=\(\")[^\"]+(\")@\1${NEW_CHECKPOINT}\2@g" "$script" || true
      rm -f "${script}.bak"
      echo "Updated checkpoint in $script"
    fi
  done
  
  # --- Update cluster constraint in YAML files ---
  for yaml_file in "$dir"/*.yaml; do
    if [ -f "$yaml_file" ]; then
      # Within the block from "constraints:" to the closing bracket "]",
      # replace any line that begins (after any whitespace) with "ai2/..."
      # with exactly two literal tabs followed by the new cluster.
      sed -i.bak -E "/constraints:/,/]/ s|^[[:space:]]*ai2/[^[:space:]]+|$(printf "\t\t")${NEW_CLUSTER}|g" "$yaml_file" || true
      rm -f "${yaml_file}.bak"
      echo "Updated cluster in $yaml_file"
    fi
  done
done

# # --- Submit experiments using beaker ---
# for dir in "${EXPERIMENT_DIRS[@]}"; do
#   echo "Submitting experiments in directory: $dir"
#   for yaml_file in "$dir"/*.yaml; do
#     if [ -f "$yaml_file" ]; then
#       echo "Running experiment for: $yaml_file"
#       beaker experiment create "$yaml_file"
#     fi
#   done
# done
