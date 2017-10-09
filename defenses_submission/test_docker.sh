cd "$( dirname "${BASH_SOURCE[0]}" )"
CODE_DIR=$(pwd)
cd ../..
HOME_DIR=$(pwd)
INPUT_DIR=${HOME_DIR}/dataset/images
OUTPUT_DIR=${CODE_DIR}/output
DOCKER_IMG="yaozhao/fatfingers:latest"

echo nvidia-docker run \
  -v "${INPUT_DIR}":/input_images \
  -v "${OUTPUT_DIR}":/output \
  -v "${CODE_DIR}":/code \
  -w /code \
  ${DOCKER_IMG} \
  ./run_defense.sh /input_images /output

nvidia-docker run \
  -v "${INPUT_DIR}":/input_images \
  -v "${OUTPUT_DIR}":/output \
  -v "${CODE_DIR}":/code \
  -w /code \
  ${DOCKER_IMG} \
  ./run_defense.sh /input_images /output/result.csv
