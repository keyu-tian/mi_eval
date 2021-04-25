REL_PATH=../../
cfg_file="${1:-"../cfg.yaml"}"

l1=$(grep -n "\[" "${cfg_file}" | cut -d ":" -f 1)
l2=$(grep -n "\]" "${cfg_file}" | cut -d ":" -f 1)
n_gpus=$(((l2-l1-1)*2))

PYTHONPATH=${PYTHONPATH}:${REL_PATH} GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --gpu -n${n_gpus} \
--cpus-per-task=6 \
--job-name MI_test \
"python -u -m main --cfg_path=\"${cfg_file}\" --dataset=liveness --train=True"
