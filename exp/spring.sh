REL_PATH=../

PYTHONPATH=${PYTHONPATH}:${REL_PATH} GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --gpu -n8 \
--job-name MI_test \
--cpus-per-task=6 \
"python -u -m main --num_classes=${2:-"50"} --n_neighbors=5 --batch_size=256 "

