#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x

PY_ARGS=${@:1}

set -o pipefail

pushd src/build
make -j4
popd

# OUTPUT_BASE=$(echo $1 | sed -e "s/configs/exps/g" | sed -e "s/.args$//g")
export DISPLAY=:0
OUTPUT_BASE=exps
mkdir -p $OUTPUT_BASE

for RUN in $(seq 100); do
  ls $OUTPUT_BASE | grep run$RUN && continue
  OUTPUT_DIR=$OUTPUT_BASE/run$RUN
  mkdir $OUTPUT_DIR && break
done

# run backup
echo "Backing up to log dir: $OUTPUT_DIR"
cp -r main.py env_gl.py rotation.py model.py $OUTPUT_DIR
echo " ...Done"

pushd $OUTPUT_DIR

# log git status
echo "Logging git status"
git status > git_status
git rev-parse HEAD > git_tag
git diff > git_diff
echo $PY_ARGS > args
echo " ...Done"

python main.py ${PY_ARGS} |& tee -a output.log
