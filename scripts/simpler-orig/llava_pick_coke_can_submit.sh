#!/bin/sh

# Define a list of names (replace with your desired names)
names="backgrounds base camera distractors lightings textures"

# Loop through each name and run the command
for name in $names; do
  echo "Running experiment for: $name"
  beaker experiment create "llava_pick_coke_can_variant_${name}.yaml"
done


# # Define a list of names (adjust as needed)
# names="backgrounds base camera distractors lightings textures"

# # Define the new cluster value you want to set.
# new_cluster="ai2/neptune-cirrascale"

# for name in $names; do
#   file="llava_pick_coke_can_variant_${name}.yaml"
  
#   # Check if the YAML file exists before modifying it.
#   if [ -f "$file" ]; then
#     echo "Updating cluster constraint in ${file}..."
#     # Use sed to replace the old cluster value with the new one.
#     # The -i.bak flag creates a backup file with .bak extension; remove it if not needed.
#     sed -i.bak "s|ai2/jupiter-cirrascale-2|${new_cluster}|g" "$file"
#     rm -f "${file}.bak"
#   else
#     echo "File ${file} not found. Skipping update."
#   fi

#   echo "Running experiment for: $name"
#   beaker experiment create "$file"
# done
