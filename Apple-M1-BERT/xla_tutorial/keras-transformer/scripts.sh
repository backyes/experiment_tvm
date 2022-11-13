# https://zhuanlan.zhihu.com/p/140343833
TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_clustering_debug" \
TF_DUMP_GRAPH_PREFIX="tmp/tf_graph"  python3 train_xla_profile.py --device=cpu
