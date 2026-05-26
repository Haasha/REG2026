#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="reg2026_algorithm"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

FIXTURES_DIR="${SCRIPT_DIR}/test/fixtures"
INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

# ---------------------------------------------------------------------------
# GPU passthrough: use --gpus all only when Docker can actually provide a GPU.
# (Avoids CDI / "no known GPU vendor" errors on Mac and other CPU-only hosts.)
# ---------------------------------------------------------------------------
DOCKER_GPU_FLAGS=()
if docker run --rm --gpus all \
    --platform=linux/amd64 \
    --entrypoint true \
    "$DOCKER_IMAGE_TAG" >/dev/null 2>&1; then
    DOCKER_GPU_FLAGS=(--gpus all)
    echo "=+= Docker GPU passthrough available (--gpus all)"
else
    echo "=+= No Docker GPU passthrough; container will run on CPU"
fi

# ---------------------------------------------------------------------------
# Cleanup: fix output permissions, remove staged inputs, remove tmp volume
# ---------------------------------------------------------------------------
cleanup() {
    echo "=+= Cleaning up ..."

    for idir in interf0 interf1; do
        if [ -d "${OUTPUT_DIR}/${idir}" ]; then
            docker run --rm \
              --platform=linux/amd64 \
              --quiet \
              --volume "${OUTPUT_DIR}/${idir}":/output \
              --entrypoint /bin/sh \
              "$DOCKER_IMAGE_TAG" \
              -c "chmod -R -f o+rwX /output/* || true"
        fi
    done

    # Remove the input dirs that were staged from fixtures
    rm -rf "${INPUT_DIR}/interf0" "${INPUT_DIR}/interf1"

    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Stage test fixtures into test/input right before running
# (test/input is otherwise kept empty in the repository)
# ---------------------------------------------------------------------------
echo "=+= Staging test fixtures into test/input"
cp -r "${FIXTURES_DIR}/interf0" "${INPUT_DIR}/interf0"
cp -r "${FIXTURES_DIR}/interf1" "${INPUT_DIR}/interf1"

# Allow the Docker user to read staged inputs and model weights
chmod -R -f o+rX "${INPUT_DIR}/interf0" "${INPUT_DIR}/interf1" "${SCRIPT_DIR}/model"

# ---------------------------------------------------------------------------
# Prepare output directories
# ---------------------------------------------------------------------------
for idir in interf0 interf1; do
    if [ -d "${OUTPUT_DIR}/${idir}" ]; then
        chmod -f o+rwX "${OUTPUT_DIR}/${idir}"
        echo "=+= Cleaning up earlier output for ${idir}"
        docker run --rm \
            --platform=linux/amd64 \
            --quiet \
            --volume "${OUTPUT_DIR}/${idir}":/output \
            --entrypoint /bin/sh \
            "$DOCKER_IMAGE_TAG" \
            -c "rm -rf /output/* || true"
    else
        mkdir -p -m o+rwX "${OUTPUT_DIR}/${idir}"
    fi
done

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

# ---------------------------------------------------------------------------
# Run a single forward pass for a given interface directory
# ---------------------------------------------------------------------------
run_docker_forward_pass() {
    local interface_dir="$1"

    echo "=+= Doing a forward pass on ${interface_dir}"

    ## Key Docker flags:
    # '--network none'             no internet access (matches Grand Challenge runtime)
    # "${DOCKER_GPU_FLAGS[@]}"     --gpus all when probe succeeded (see above)
    # '--volume <NOOP>:/tmp'       /tmp is ephemeral on the platform — mirrored here
    # '--volume ./model:/opt/ml/model:ro'  local model weights mount
    docker run --rm "${DOCKER_GPU_FLAGS[@]}" \
        --platform=linux/amd64 \
        --network none \
        --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
        --volume "${OUTPUT_DIR}/${interface_dir}":/output \
        --volume "$DOCKER_NOOP_VOLUME":/tmp \
        --volume "${SCRIPT_DIR}/model":/opt/ml/model:ro \
        "$DOCKER_IMAGE_TAG"

    echo "=+= Results written to ${OUTPUT_DIR}/${interface_dir}"
}

# ---------------------------------------------------------------------------
# Run both interfaces
# ---------------------------------------------------------------------------
run_docker_forward_pass "interf0"
run_docker_forward_pass "interf1"

python3 "${SCRIPT_DIR}/test/validate_outputs.py"

echo "=+= All done. Save for upload with ./do_save.sh"
