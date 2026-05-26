#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="reg2026_algorithm"

echo ""
echo "= STEP 1 = (Re)build the image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"
echo "==== Done"
echo ""

# Derive a timestamp from the image metadata
build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG" )

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for $DOCKER_IMAGE_TAG"
    exit 1
fi

formatted_build_info=$(echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')
output_filename="${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
output_path="${SCRIPT_DIR}/${output_filename}"

# Save the Docker image as a gzipped tarball
echo "= STEP 2 = Saving the image (this may take a while)"
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$output_path"
printf "Saved as: \e[32m%s\e[0m\n" "$output_filename"
echo "==== Done"
echo ""

# Pack the model weights directory
echo "= STEP 3 = Packing model weights (this may take a while)"
output_tarball_name="${SCRIPT_DIR}/model.tar.gz"
tar -czf "$output_tarball_name" -C "${SCRIPT_DIR}/model" .
printf "Saved as: \e[32mmodel.tar.gz\e[0m\n"
echo "==== Done"
echo ""

printf "\e[33mNext steps:\e[0m\n"
printf "  1. Upload \e[32m%s\e[0m  →  Grand Challenge > Algorithm > Container images\n" "$output_filename"
printf "  2. Upload \e[32mmodel.tar.gz\e[0m           →  Grand Challenge > Algorithm > Models\n"
