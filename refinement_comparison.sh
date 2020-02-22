python pose2motion.py \
  --exp_name h36_gt_v4 \
  --batch_size 32 --lr_decay 12800 --lr 1e-3 \
  --past 8 --future 20 --refine_version 4 --refine_iteration 1 \
  --time_stride 2 --window_stride 2 \
  --visible_device 1 --final True \
  --keypoint_source gt --dropout 0.25

python pose2motion.py \
  --exp_name h36_gt_v3 \
  --batch_size 32 --lr_decay 12800 --lr 1e-3 \
  --past 8 --future 20 --refine_version 3 --refine_iteration 1 \
  --time_stride 2 --window_stride 2 \
  --visible_device 1 --final True \
  --keypoint_source gt --dropout 0.25

python pose2motion.py \
  --exp_name h36_gt_v2 \
  --batch_size 32 --lr_decay 12800 --lr 1e-3 \
  --past 8 --future 20 --refine_version 2 --refine_iteration 1 \
  --time_stride 2 --window_stride 2 \
  --visible_device 1 --final True \
  --keypoint_source gt --dropout 0.25

python pose2motion.py \
  --exp_name h36_gt_v1 \
  --batch_size 32 --lr_decay 12800 --lr 1e-3 \
  --past 8 --future 20 --refine_version 1 --refine_iteration 1 \
  --time_stride 2 --window_stride 2 \
  --visible_device 1 --final True \
  --keypoint_source gt --dropout 0.25
