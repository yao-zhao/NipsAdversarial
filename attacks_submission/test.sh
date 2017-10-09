cd "$( dirname "${BASH_SOURCE[0]}" )"
CODE_DIR=$(pwd)
OUTPUT_DIR=${CODE_DIR}/output_images

if [ ! -d $OUTPUT_DIR ]; then
  mkdir $OUTPUT_DIR
else
  rm -f "${OUTPUT_DIR}/*.png"
fi
cd ../..
HOME_DIR=$(pwd)
INPUT_DIR=${HOME_DIR}/dataset/images
MAX_EPSILON=16

cd ${CODE_DIR}
python ${CODE_DIR}/attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}"
