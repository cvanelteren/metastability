#!/bin/sh
# setup experiments
source activate base

file="./tasks.txt"
while IFS= read -r line;
do
    # sbatch -W ./single_task.sh "$line"
    sh single_task.sh "$line"
done < "$file"
