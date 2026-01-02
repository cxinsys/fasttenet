python tutorial_notf.py\
 --fp_exp expression_dataTuck_sub.csv\
 --fp_trj pseudotimeTuck.txt\
 --fp_br cell_selectTuck.txt\
 --backend torch\
 --num_devices 8\
 --batch_size 32768\
 --sp_rm TE_result_matrix.txt

 python reconstruct_grn.py \
  --fp_rm TE_result_matrix.txt \
  --fdr 0.01 \
  --backend gpu \
  --device_id 0