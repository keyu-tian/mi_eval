REL_PATH=../

PYTHONPATH=${PYTHONPATH}:${REL_PATH} GLOG_vmodule=MemcachedClient=-1 \
spring.submit run -r --gpu -n8 \
--cpus-per-task=6 \
--job-name MI_test \
"python -u -m main --num_classes=${2:-"50"} --n_neighbors=5 --batch_size=256 "

