# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth



declare -a ckpt_paths=("hqfang/random-half-step37222-hf")

for scene_name in "${scene_names[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    for env_name in "${env_names[@]}"; do
      EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt model_ids=apple"
      EvalSim
    done
  done
done


# # lightings
# scene_name=frl_apartment_stage_simple

# for ckpt_path in "${ckpt_paths[@]}"; do
#   for env_name in "${env_names[@]}"; do
#     EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=brighter model_ids=apple"
#     EvalSim
#     EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=darker model_ids=apple"
#     EvalSim
#   done
# done


# # new cabinets
# scene_name=frl_apartment_stage_simple

# for ckpt_path in "${ckpt_paths[@]}"; do
#   for env_name in "${env_names[@]}"; do
#     EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station2 model_ids=apple"
#     EvalSim
#     EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station3 model_ids=apple"
#     EvalSim
#   done
# done