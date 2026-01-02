python tutorial_tf.py\
 --fp_exp skin_exp_data.csv\
 --fp_trj skin_pseudotime.txt\
 --fp_br skin_cellselect.txt\
 --fp_tf human_tfs.txt\
 --backend torch\
 --num_devices 8\
 --batch_size 2048\
 --sp_rm TE_result_matrix.txt

  python reconstruct_grn.py \
  --fp_rm TE_result_matrix.txt \
  --fdr 0.01 \
  --fp_tf human_tfs.txt\
  --backend gpu \
  --device_id 0