REL_PATH=../../

PYTHONPATH=${PYTHONPATH}:${REL_PATH} GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --gpu -n8 \
--cpus-per-task=6 \
--job-name MI_test \
"python -u -m main --dataset=liveness --num_classes=2 --n_neighbors=${1:-"10"} --batch_size=256 "

