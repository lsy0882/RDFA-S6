#!/bin/bash

# Define an array with the directory names to exclude.
exclude_dirs=("b2_me50_ResidualSharedBiMambaBackbone_v1.11.0-104" "b2_me50_ResidualSharedBiMambaBackbone_v1.11.0-94" "b2_me50_ResidualSharedBiMambaBackbone_v1.11.0-89" "b2_me50_ResidualSharedBiMambaBackbone_v1.11.0-81" "b2_me50_ResidualSharedBiMambaBackbone_v1.11.0-66" "b2_me50_ResidualSharedBiMambaBackbone_v1.11.0-59" "b2_me50_ResidualSharedBiMambaBackbone_v1.11.0-31" "b2_me50_ResidualSharedBiMambaBackbone_v1.11.0-21" "b2_me50_ResidualSharedBiMambaBackbone_v1.11.0-19" "b2_me50_ResidualSharedBiMambaBackbone_v1.11.0-4" "b2_me50_ResidualSharedBiMambaBackbone_v1.9.0-2" "b2_me50_ResidualSharedBiMambaBackbone_v1.8.0-3" "b2_me50_ResidualSharedBiMambaBackbone_v1.7.0-1" "b2_me50_ResidualSharedBiMambaBackbone_v1.4.0-17" "b2_me50_ResidualSharedBiMambaBackbone_v1.2.5")

# Generate the -name option to use with the find command.
exclude_find_opts=""
for dir in "${exclude_dirs[@]}"; do
    exclude_find_opts+="! -name '$dir' "
done

# Execute the find command to delete directories except for the excluded ones.
eval "find . -maxdepth 1 -type d $exclude_find_opts ! -name '.' -exec rm -r {} +"

echo "Specified directories excluded and others deleted."