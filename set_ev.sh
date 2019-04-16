# !/bin/bash
#CODE_ROOT=/home/psdz/works/dataArea/share/code/gxh
#MODELS_DIR=/home/psdz/works/tf/tensorflow/tensorflow/models
CODE_ROOT=`pwd`
MODELS_DIR=$CODE_ROOT/models-master
echo $CODE_ROOT
############################################################
LIB_DIR=$CODE_ROOT/lib #$CODE_ROOT/../gxh/lib
echo $LIB_DIR
#export LD_LIBRARY_PATH=`pwd`:$LIB_DIR
############################################################
if [ -f "$LIB_DIR/libcusolver.so.9.0" ];then
echo "exist"
else
echo "create file by ln"
ln -s $LIB_DIR/libcusolver.so.9.0.103 $LIB_DIR/libcusolver.so.9.0
fi

if [ -f "$LIB_DIR/libcudart.so.9.0" ];then
echo "exist"
else
echo "create file by ln"
ln -s $LIB_DIR/libcudart.so.9.0.103 $LIB_DIR/libcudart.so.9.0
fi

if [ -f "$LIB_DIR/libcurand.so.9.0" ];then
echo "exist"
else
echo "create file by ln"
ln -s $LIB_DIR/libcurand.so.9.0.103 $LIB_DIR/libcurand.so.9.0
fi
#ln -s $LIB_DIR/libcusolver.so.9.0.103 $LIB_DIR/libcusolver.so.9.0
#ln -s $LIB_DIR/libcudart.so.9.0.103 $LIB_DIR/libcudart.so.9.0
#ln -s $LIB_DIR/libcurand.so.9.0.103 $LIB_DIR/libcurand.so.9.0
############################################################

SRC_DIR=$MODELS_DIR/research
export PYTHONPATH=$PYTHONPATH:$SRC_DIR
export PYTHONPATH=$PYTHONPATH:$SRC_DIR/slim
#../../protoc-3.3.0/bin/protoc object_detection/protos/*.proto --python_out=.
$CODE_ROOT/protoc-3.3.0/bin/protoc -I=$SRC_DIR \
$SRC_DIR/object_detection/protos/*.proto \
--proto_path=$SRC_DIR/object_detection/protos \
--python_out=$SRC_DIR
python $SRC_DIR/object_detection/builders/model_builder_test.py

#CUDA_VISIBLE_DEVICES=0 python mytest2.py
CUDA_VISIBLE_DEVICES=0 python mytest2-2.py
#locate *.ttc
echo "20190318"

