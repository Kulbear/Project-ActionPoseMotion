from .pose_motion_model import (
    PoseLifter,
    MotionGenerator,
    Pose2MotNet
)

from .refinement_model import (
    TrajRefinementModule,
    ResTrajRefinementModule,
    ResConvTrajRefinementModule,
    EnsTrajRefinementModule
)

from .keypoint_refinement import (
    KeypointRefineNet
)

REFINEMENT_ARCHS = {
    1: TrajRefinementModule,
    2: ResTrajRefinementModule,
    3: ResConvTrajRefinementModule,
    4: EnsTrajRefinementModule
}
