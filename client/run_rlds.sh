conda run --no-capture-output -n openvla_tsc python /home/iflab-zzh-intern/tiansicheng/openvla/piper/hdf5_to_rlds.py \
  --input_dir "/home/iflab-zzh-intern/tiansicheng/openvla/piper/data/pick up the banana2" \
  --output_dir "/home/iflab-zzh-intern/tiansicheng/openvla/piper/data_rlds_opt" \
  --dataset_name "pick_up_the_banana2" \
  --version "1.0.0" \
  --exclude_fail \
  --min_action_norm 0