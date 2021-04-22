REL_PATH=../../

PYTHONPATH=${PYTHONPATH}:${REL_PATH} GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --gpu -n16 \
--cpus-per-task=6 \
--job-name MI_test \
"python -u -m main"
