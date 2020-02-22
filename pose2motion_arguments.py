import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', required=True, type=str,
                        help='The name of the experiment.')
    parser.add_argument('--final', default=False, type=bool,
                        help='Whether do the final result evaluation (accurate pose and motion)')

    parser.add_argument('--dataset', default='h36m', type=str,
                        help='Target experiment dataset.')
    parser.add_argument('--keypoint_source', default='gt', type=str,
                        help='2D detections to use. Could be one of "gt", "sh_ft" and "sh_pt_mpii".')
    parser.add_argument('--actions', default='*', type=str,
                        help='Actions to train/test on, separated by comma, or * for all.')

    # dataset and dataloader
    parser.add_argument('--past', default=8, type=int,
                        help='The length of the (observed) past pose sequence that we want to lift from 2D to 3D.')
    parser.add_argument('--future', default=16, type=int,
                        help='The length of the future motion sequence that we want to generate.')
    parser.add_argument('--time_stride', default=1, type=int,
                        help='The stride size used for (possibly) downsampling the frame rate. '
                             'For example, time_stride = 2 means we down sample 30 fps to 15 fps.')
    parser.add_argument('--window_stride', default=4, type=int,
                        help='The stride size used for generating pose sequence.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Dare you don\'t know what this is? Hah!')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='The number of workers used for the dataloader.')

    # model
    parser.add_argument('--encoder_ipt_dim', default=32, type=int,
                        help='The input dimension of the pose encoder.')
    parser.add_argument('--encoder_opt_dim', default=45, type=int,
                        help='The output dimension of the pose encoder.')
    parser.add_argument('--decoder_ipt_dim', default=45, type=int,
                        help='The input dimension of the motion decoder.')
    parser.add_argument('--decoder_opt_dim', default=45, type=int,
                        help='The output dimension of the motion decoder.')
    parser.add_argument('--num_recurrent_layers', default=2, type=int,
                        help='The number of rnn/lstm/gru layers for the pose encoder and the motion decoder.')
    parser.add_argument('--bidirectional', default=True, type=bool,
                        help='Whether to use bidirectional recurrent layer.')
    parser.add_argument('--hid_dim', default=256, type=int,
                        help='The number of hidden dimensions in the encoder and the decoder.')
    parser.add_argument('--dropout', default=0., type=float,
                        help='Dropout rate used in all dropout layers.')

    # training
    parser.add_argument('--visible_devices', default='0', type=str,
                        help='Visible device for using.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device used for training.')
    parser.add_argument('--lr', default=2e-3, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_decay', type=int, default=10000,
                        help='The number of iterations that we perform learning rate decay.')
    parser.add_argument('--lr_gamma', type=float, default=0.9,
                        help='The rate of learning rate decay.')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Starting epoch number.')
    parser.add_argument('--epochs', default=36, type=int,
                        help='The number of training epochs.')

    # evaluation
    parser.add_argument('--ckpt_path', default=None, type=str,
                        help='The path to the trained model.')
    parser.add_argument('--evaluation', default=False, type=bool,
                        help='Evaluation only mode.')

    # visualization
    parser.add_argument('--viz_subject', default='S1', type=str,
                        help='The subject used in visualization.')
    parser.add_argument('--viz_target', default='past', type=str,
                        help='The visualization target, can be one of "past" or "future".')
    parser.add_argument('--viz_camera', default=1, type=int,
                        help='The camera used in visualization.')
    parser.add_argument('--viz_action', default='Walking 1', type=str,
                        help='The action used in visualization.')
    parser.add_argument('--viz_video', type=str, default=None,
                        help='path to input video')
    parser.add_argument('--viz_skip', type=int, default=0,
                        help='skip first N frames of input video')
    parser.add_argument('--viz_output', type=str, default='vis.mp4',
                        help='output file name (.gif or .mp4)')
    parser.add_argument('--viz_bitrate', type=int, default=3000,
                        help='bitrate for mp4 videos')
    parser.add_argument('--viz_limit', type=int, default=500,
                        help='only render first N frames')
    parser.add_argument('--viz_downsample', type=int, default=1,
                        help='downsample FPS by a factor N')
    parser.add_argument('--viz_size', type=int, default=5,
                        help='image size')

    parser.add_argument('--refine_version', type=int, default=1,
                        help='The refine model version used.')
    parser.add_argument('--refine_iteration', type=int, default=1,
                        help="The number of refinement performed.")

    args = parser.parse_args()
    return args
