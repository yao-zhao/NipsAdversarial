cd "$( dirname "${BASH_SOURCE[0]}" )"
CODE_DIR=$(pwd)
cd ../..
HOME_DIR=$(pwd)
INPUT_DIR=${HOME_DIR}/dataset/images
OUTPUT_DIR=${HOME_DIR}/dataset/output_images
DOCKER_IMG="yaozhao/fatfingers:latest"
MAX_EPSILON=16

echo nvidia-docker run \
  -v "${INPUT_DIR}":/input_images \
  -v "${OUTPUT_DIR}":/output_images \
  -v "${CODE_DIR}":/code \
  -w /code \
  ${DOCKER_IMG} \
  ./run_attack.sh /input_images /output_images 16

nvidia-docker run \
  -v "${INPUT_DIR}":/input_images \
  -v "${OUTPUT_DIR}":/output_images \
  -v "${CODE_DIR}":/code \
  -w /code \
  ${DOCKER_IMG} \
  ./run_attack.sh /input_images /output_images 16
