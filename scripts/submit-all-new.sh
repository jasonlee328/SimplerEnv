#!/bin/bash
# Hard-coded checkpoint and cluster values
NEW_CHECKPOINT="molmo-act/apr-14-float-ratio"
NEW_POLICY_MODEL="--policy-model molmo"
CUSTOM_YAML_ARGS="nvidia-smi &&
apt update && apt install libglvnd-dev libvulkan1 libjpeg-dev libpng-dev libglib2.0-0 ffmpeg -y &&
mkdir -p /data/input/jiafei/GroundedVLA/ &&
cd /data/input/jiafei/GroundedVLA && pwd &&
git clone --recursive https://github.com/jasonlee328/SimplerEnv.git &&
source /opt/conda/etc/profile.d/conda.sh &&
conda activate simpler_env &&
cd /data/input/jiafei/GroundedVLA/SimplerEnv/ManiSkill2_real2sim &&
pip install -e. &&
cd /data/input/jiafei/GroundedVLA/SimplerEnv && 
pip install -e . &&
pip install transformers==4.49.0 &&
pip install torch==2.6.0 &&
pip install --no-cache-dir --upgrade torchvision &&
pip install --no-cache-dir --upgrade flash-attn &&
pip install accelerate==1.6.0 &&"

NEW_YAML_CLUSTER_ARG="ai2/augusta-google-1"
NEW_PRIORITY="urgent"

EXPERIMENT_DIRS=(
  "/data/input/jiafei/SimplerEnv/scripts/pick_coke_can"
  "/data/input/jiafei/SimplerEnv/scripts/open_drawer"
  "/data/input/jiafei/SimplerEnv/scripts/move_near"
)

for dir in "${EXPERIMENT_DIRS[@]}"; do
  echo "Processing directory: $dir"
  
  # --- Update checkpoint paths and policy model in bash scripts ---
  for script in "$dir"/*.sh; do
    if [ -f "$script" ]; then
      # Update the single-line array for "arr"
      sed -i.bak -E "s@^(declare -a arr=\(\")[^\"]+(\")@\1${NEW_CHECKPOINT}\2@g" "$script" || true
      
      # Update the single-line definition of ckpt_paths
      sed -i.bak -E "s@^(declare -a ckpt_paths=\(\")[^\"]+(\")@\1${NEW_CHECKPOINT}\2@g" "$script" || true
      
      # Update multi-line definition of ckpt_paths if present
      sed -i.bak -E '/^declare -a ckpt_paths=\([^)]*$/,/^[[:space:]]*\)[[:space:]]*$/c\declare -a ckpt_paths=("'"${NEW_CHECKPOINT}"'")' "$script" || true
      
      # Update policy model in bash scripts
      sed -i.bak -E "s/--policy-model[[:space:]]+[^[:space:]]+/${NEW_POLICY_MODEL}/g" "$script" || true
      
      rm -f "${script}.bak"
      echo "Updated checkpoint and policy model in $script"
    fi
  done
  
  # --- Update YAML files ---
  for yaml_file in "$dir"/*.yaml; do
    if [ -f "$yaml_file" ]; then
      # 1. Update the cluster block inside the constraints.
      #    Look for a line starting with "cluster:" that contains an opening "[".
      #    If the closing bracket is on the same line, substitute inline.
      #    Otherwise, replace the entire multi-line block with a single line containing the new cluster.
      awk -v newcluster="$NEW_YAML_CLUSTER_ARG" '
      /^[[:space:]]*cluster:[[:space:]]*\[/ {
          # Capture leading whitespace.
          match($0, /^[[:space:]]*/);
          indent = substr($0, RSTART, RLENGTH);
          if ($0 ~ /\]/) {
              # Single-line cluster definition: substitute content between [ and ].
              gsub(/\[[^]]*\]/, "[ " newcluster " ]");
              print;
          } else {
              # Multi-line cluster: print the new cluster line and enter skip mode.
              print indent "cluster: [ " newcluster " ]";
              inCluster = 1;
          }
          next;
      }
      inCluster {
          # Look for the closing bracket in a multi-line cluster block.
          if ($0 ~ /\]/) { inCluster = 0; }
          # Skip all lines inside the cluster block.
          next;
      }
      { print }
      ' "$yaml_file" > "$yaml_file.tmp" && mv "$yaml_file.tmp" "$yaml_file"

      echo "Updated cluster block in $yaml_file"
      
      # --- Update only the YAML arguments block ---
      # This awk block:
      # 1. Finds the "arguments:" key and calculates its indentation.
      # 2. Replaces the current indented arguments block with:
      #    - A "- >-" line.
      #    - The custom commands from CUSTOM_YAML_ARGS.
      #    - An extra command line built from the current YAML file path (with .yaml replaced by .sh).
      # 3. Skips over the original arguments block while preserving subsequent lines.
      awk -v newargs="$CUSTOM_YAML_ARGS" -v yamlpath="$yaml_file" '
      BEGIN { skip = 0; argsIndent = 0 }
      # Detect the "arguments:" line
      $0 ~ /^[[:space:]]*arguments:/ {
          # Compute the indentation of the current line by removing all non-space characters.
          indent = $0; gsub(/[^[:space:]]/, "", indent);
          argsIndent = length(indent);
          print $0;
          # Define block indentation: two extra spaces for the "- >-" line and four for command lines.
          indentBlock = indent "";
          indentContent = indent "  ";
          print indentBlock "- >-";
          n = split(newargs, arrArgs, "\n");
          for (i = 1; i <= n; i++) {
              print indentContent arrArgs[i];
          }
          # Generate the extra command line: replace .yaml with .sh in the full YAML path.
          newpath = yamlpath; sub(/\.yaml$/, ".sh", newpath);
          print indentContent "bash " newpath;
          skip = 1;
          next;
      }
      # Skip lines that are part of the original arguments block (i.e. indented more than the "arguments:" key)
      skip == 1 {
          indent = $0; gsub(/[^[:space:]]/, "", indent);
          currentIndent = length(indent);
          if (currentIndent > argsIndent) {
              next;
          } else {
              skip = 0;
          }
      }
      { print }
      ' "$yaml_file" > "$yaml_file.new" && mv "$yaml_file.new" "$yaml_file"
      
      echo "Updated YAML arguments in $yaml_file"
      # 3. Update the context priority value
      awk -v new_priority="$NEW_PRIORITY" '
      /^[[:space:]]*context:/ {
          print;
          in_context = 1;
          next;
      }
      in_context && /^[[:space:]]*priority:/ {
          sub(/priority:[[:space:]]*[^\n]+/, "priority: " new_priority);
          in_context = 0;  # update only one line
      }
      { print }
      ' "$yaml_file" > "$yaml_file.tmp" && mv "$yaml_file.tmp" "$yaml_file"

      echo "Updated YAML context priority in $yaml_file"
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








#!/bin/bash

EXPERIMENT_DIRS=(
  "/data/input/jiafei/GroundedVLA/SimplerEnv/scripts/pick_coke_can"
  "/data/input/jiafei/GroundedVLA/SimplerEnv/scripts/in_drawer"
  "/data/input/jiafei/GroundedVLA/SimplerEnv/scripts/move_near"
  "/data/input/jiafei/GroundedVLA/SimplerEnv/scripts/open_drawer"
)

echo "Counting .sh files in each directory:"

for dir in "${EXPERIMENT_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    count=$(find "$dir" -maxdepth 1 -type f -name "*.sh" | wc -l)
    echo "$dir: $count .sh file(s)"
  else
    echo "$dir: Directory does not exist"
  fi
done
