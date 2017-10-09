cd "$( dirname "${BASH_SOURCE[0]}" )"
CODE_DIR=$(pwd)
OUTPUT_DIR=${CODE_DIR}/output.csv

if [ ! -f $OUTPUT_DIR ]; then
  touch $OUTPUT_DIR
fi

cd ../..
HOME_DIR=$(pwd)
INPUT_DIR=${HOME_DIR}/dataset/images

cd ${CODE_DIR}
python ${CODE_DIR}/defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_DIR}"
