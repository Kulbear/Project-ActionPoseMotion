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
    KeypointRefineNetV1,
    KeypointRefineNetV2
)

PRE_REFINEMENT_ARCHS = {
    1: KeypointRefineNetV1,
    2: KeypointRefineNetV2
}

REFINEMENT_ARCHS = {
    1: TrajRefinementModule,
    2: ResTrajRefinementModule,
    3: ResConvTrajRefinementModule,
    4: EnsTrajRefinementModule
}
