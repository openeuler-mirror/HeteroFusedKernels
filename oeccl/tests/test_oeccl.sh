source env.sh
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_GLOBAL_EVENT_ENABLE=0
#export ASCEND_PROCESS_LOG_PATH=./log
export MASTER_ADDR="localhost"
export MASTER_PORT=60087
export WORLD_SIZE=8
export PROTCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ifmsprof=0
export PYTHONUNBUFFERED=1
#export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OECCL_NUMA_MAP="0,0,0,0,1,1,1,1"
rm -rf /dev/shm/*
rm -rf /hugepages/*

python test_oeccl.py
