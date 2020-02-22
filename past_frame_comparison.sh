python pose2motion.py \
  --exp_name h36_gt_v2_p81 \
  --batch_size 32 --lr_decay 12800 --lr 1e-3 \
  --past 81 --future 20 --refine_version 2 --refine_iteration 1 \
  --time_stride 2 --window_stride 2 \
  --visible_device 0 --final True \
  --keypoint_source gt --dropout 0.25

#python pose2motion.py \
#  --exp_name h36_gt_v2_p243 \
#  --batch_size 32 --lr_decay 2000 --lr 1e-3 \
#  --past 243 --future 20 --refine_version 2 --refine_iteration 1 \
#  --time_stride 2 --window_stride 20 \
#  --visible_device 0 --final True \
#  --keypoint_source gt --dropout 0.25

python pose2motion.py \
  --exp_name h36_gt_v2_p27 \
  --batch_size 32 --lr_decay 12800 --lr 1e-3 \
  --past 27 --future 20 --refine_version 2 --refine_iteration 1 \
  --time_stride 2 --window_stride 2 \
  --visible_device 0 --final True \
  --keypoint_source gt --dropout 0.25

python pose2motion.py \
  --exp_name h36_gt_v2_p5 \
  --batch_size 32 --lr_decay 12800 --lr 1e-3 \
  --past 5 --future 20 --refine_version 2 --refine_iteration 1 \
  --time_stride 2 --window_stride 2 \
  --visible_device 0 --final True \
  --keypoint_source gt --dropout 0.25