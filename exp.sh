#!/bin/bash

# Directory paths
REPO_DIR=$(pwd)
BUILD_DIR="${REPO_DIR}/build"
INPUT_IMAGE="${REPO_DIR}/data/tower.jpg"
OUTPUT_DIR="${REPO_DIR}/outputs"

# Clean and build the project
echo "Cleaning and building the project..."
make clean
make all
if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Define seam numbers
SEAM_NUMS=(10 50 150)

# Define target sizes
TARGET_SIZES=("800x600" "400x300")

# Define modes
declare -A modes
modes=(
    ["remove_vertical"]=" "
    ["remove_horizontal"]="--horizontal "
    ["insert_vertical"]="--insert "
    ["insert_horizontal"]="--insert --horizontal "
    ["target_resize_1"]="--target ${TARGET_SIZES[0]} "
    ["target_resize_2"]="--target ${TARGET_SIZES[1]} "
)

# Run experiments
for mode_name in "${!modes[@]}"; do
    mode_options="${modes[${mode_name}]}"
    
    if [[ ${mode_name} == "target_resize_"* ]]; then
        # Handle target resize experiments
        output_file="${OUTPUT_DIR}/output_${mode_name}.png"
        command="./seam_carving -i ${INPUT_IMAGE} -o ${output_file} ${mode_options}"
        echo "Executing: ${command}"
        # Run the command in the build directory
        cd ${BUILD_DIR} && ./seam_carving -i ${INPUT_IMAGE} -o ${output_file} ${mode_options}
        if [ $? -ne 0 ]; then
            echo "Experiment ${mode_name} failed."
        else
            echo "Output saved to ${output_file}"
        fi
        cd ${REPO_DIR}
    else
        # Handle seam removal/insertion experiments
        for seam_num in "${SEAM_NUMS[@]}"; do
            # Construct output filename
            output_file="${OUTPUT_DIR}/output_${mode_name}_n${seam_num}.png"
            # Construct command
            command="./seam_carving -i ${INPUT_IMAGE} -o ${output_file} -n ${seam_num} ${mode_options}"
            echo "Executing: ${command}"
            # Run the command in the build directory
            cd ${BUILD_DIR} && ./seam_carving -i ${INPUT_IMAGE} -o ${output_file} -n ${seam_num} ${mode_options}
            if [ $? -ne 0 ]; then
                echo "Experiment ${mode_name} with seam number ${seam_num} failed."
            else
                echo "Output saved to ${output_file}"
            fi
            cd ${REPO_DIR}
        done
    fi
done

echo "All experiments completed."