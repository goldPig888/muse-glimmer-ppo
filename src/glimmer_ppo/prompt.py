"""System prompt for PPO episodes.

Copied from ``llm-agent-memory/scripts/run_harness.py::prompt`` (wrist-v3
profile, commit 48989e2) so training sees the same instructions as the
zero-shot harness trials. Only the decision/step limits are parameterized.
Keep the two in sync, or the RL result is not comparable to the ablations.
"""
from llm_agent_memory.harness.experiments import image_instructions
from llm_agent_memory.harness.upstream import SYSTEM_PROMPT


def system_prompt(max_decisions: int = 80, max_steps: int = 900, clean_images: bool = True) -> str:
    text = SYSTEM_PROMPT.replace('Gemini 4 Argon controlling a real 6-DoF YAM robot arm',
                                 'Muse Glimmer controlling a simulated 7-DoF Panda robot arm')
    text = text.replace('A human operator previews every motion segment in 3D and must approve it\nbefore the arm moves. If they reject, their feedback is in the response --\nfollow it.',
                        'An automatic Panda IK and swept-contact checker previews each segment.\nRejected segments return IK_FAILURE or COLLISION; correct the proposal.')
    text = text.replace('Arm base frame: +x forward (away from the arm base), +y left, +z up.',
                        'LIBERO world frame: x and y follow the calibrated grid, +z up.')
    text = text.replace('Overhead camera (calibrated)', 'Fixed agentview camera (calibrated; oblique rather than overhead)')
    text = text.replace('from stereo depth (noisy, ~1-2 cm)', 'from the simulated camera depth sensor')
    text = text.replace('deproject through the overhead image only.',
                        'choose camera="agentview" for coarse targeting or camera="wrist" for fine targeting.')
    text = text.replace('of the LAST overhead image', 'of the selected camera image')
    text = text.replace('approach_height=0.08', 'approach_height=0.05')
    text = text.replace('Stay inside obs.workspace.', 'Stay inside x,y in [-0.6,0.6], z in [table_z,1.6].')
    text += '''\n## Two-stage Panda targeting
Agentview is oblique; a wrong height plane shifts world x,y substantially.
Coarse: choose the requested object carefully, then use
x,y,z=robot.deproject_depth(u=...,v=...,camera="agentview")
and robot.goto(x=x,y=y,z=z+0.13,pitch_deg=90,yaw_deg=0).
Depth returns the visible surface, not an object's center. The first move is
only a hover: re-observe before descending. Confirm the actual wrist view is
top-down and the object is visible; do not assume this for a tilted hand.
Fine: click the object's top center in the full wrist image, then use
x,y=robot.deproject(u=...,v=...,z_plane=...,camera="wrist").
Each camera's normalized u,v are in its displayed image (top-left origin).
Only agentview is cropped by look_at; wrist stays full-frame.
Use the correct grasp height and bracket the object between the jaws.
robot.grasp accepts support_z: record the support surface when known.
After lift, visually verify object motion; holding alone is only a proxy.
robot.place_relative(x=...,y=...,support_z=...,release_clearance=0.01,
 yaw_deg=...,approach_height=0.05) uses the retained grasp's measured height
offset. The destination support height must be chosen from observation; it
does not infer basket floors, rim heights, or objects from hidden poses.
At most two consecutive zoom-only plans are allowed. Choose a target and
propose a hover or corrected action after that; repeated zooms waste calls.
Use steps_remaining to plan: contact descent is slow only in its last 2 cm.
'''
    start = text.index('- Reach:')
    end = text.index('- After a grasp', start)
    text = text[:start] + text[end:]
    text += f'''\n## LIBERO adaptation\nThis is a single arm: only robot.<primitive> is valid. No left/right or parallel calls.
The shell command is only a response format: it is parsed, never executed.
At most {max_decisions} model decisions and {max_steps} simulator steps are available. Each primitive
may consume many steps; the simulator checks task success after every step.
Gripper command is 0=open, 1=close; finger_positions are measured in meters.
holding is only a finger-gap estimate, not proof of an object grasp: verify visually
after lifting. Only the automatic LIBERO predicate establishes task success.
Use measured poses and tracking errors. Closed fingers on nothing are not success.
Grid is 2 cm; labels give world x,y in meters. Both images are upright 512x512.
Camera pixel coordinates are normalized to the selected displayed image. Agentview is cropped; wrist is full-frame.
The table_z and workspace bounds are in the observation, not zero-height coordinates.
Keep # Thought short (40 words). On turn one include # Plan and granularity.
Return one complete curl /obs or curl /plan with a complete heredoc. Do not return
JSON, markdown code fences, unrelated shell commands, or simulator internals.
'''
    return image_instructions(text, clean_images)
