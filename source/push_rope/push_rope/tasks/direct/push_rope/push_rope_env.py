# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform
from isaaclab.assets import RigidObject
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdPhysics, Gf, Sdf

from isaaclab.markers import VisualizationMarkers

import numpy as np

from .push_rope_env_cfg import PushRopeEnvCfg


class PushRopeEnv(DirectRLEnv):
    cfg: PushRopeEnvCfg

    def __init__(self, cfg: PushRopeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.first_segment_target_pos = torch.tensor([-0.3, 0.3, 0.8], device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0

        self.rope_target_pos = torch.zeros((self.num_envs, self.cfg.num_segments, 2), dtype=torch.float, device=self.device)

    def _setup_scene(self):
        self.table = RigidObject(cfg=self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table

        # agent
        self.pusher = RigidObject(cfg=self.cfg.pusher)
        self.scene.rigid_objects["pusher"] = self.pusher

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
    
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # CREATE ROPE
        segment_length = self.cfg.segment_length
        num_segments = self.cfg.num_segments
        self.rope_rigid_bodies = []
        self.rope_rigid_bodies_prim_path = []
        self.joints = []
        damping = self.cfg.damping
        stiffness = self.cfg.stiffness

        rope_starting_point = Gf.Vec3f(-0.3, 0.0, 0.8)
        segment_prim_path = self.cfg.segment_cfg.prim_path
        # Create all rope segments
        for seg_ind in range(num_segments):
            x = rope_starting_point[0] + seg_ind * (segment_length)  # Start from rope_starting_point's x coordinate
            position = Gf.Vec3f(x, rope_starting_point[1], rope_starting_point[2])
            self.cfg.segment_cfg.prim_path = segment_prim_path + str(seg_ind)
            self.cfg.segment_cfg.init_state.pos = (position[0], position[1], position[2])

            self.rope_rigid_bodies.append(RigidObject(cfg=self.cfg.segment_cfg))
            self.scene.rigid_objects[f"rope_segment_{seg_ind}"] = self.rope_rigid_bodies[-1]

            self.rope_rigid_bodies_prim_path.append(self.cfg.segment_cfg.prim_path)

    
        for i in range(self.cfg.scene.num_envs):            
            # Create D6 joints between consecutive segments
            joint_x = (segment_length)/2.0
            for seg_ind in range(num_segments - 1):
                joint_path = Sdf.Path(f"/World/envs/env_{i}/Joint_{seg_ind}")
                
                body0_path = Sdf.Path(f"/World/envs/env_{i}/Segment_{seg_ind}")
                body1_path = Sdf.Path(f"/World/envs/env_{i}/Segment_{seg_ind + 1}")
                
                local_pos0 = Gf.Vec3f(joint_x, 0, 0)
                local_pos1 = Gf.Vec3f(-joint_x, 0, 0)
                
                # Create a joint
                joint = UsdPhysics.Joint.Define(get_current_stage(), joint_path)
                joint_prim = joint.GetPrim()

                # Lock all translational degrees of freedom
                for axis in ["transX", "transY", "transZ", "rotX"]:
                    limitAPI = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
                    limitAPI.CreateLowAttr(1.0)
                    limitAPI.CreateHighAttr(-1.0)

                # Set the bodies to connect
                joint.GetBody0Rel().SetTargets([body0_path])
                joint.GetBody1Rel().SetTargets([body1_path])

                # Set local positions
                joint.CreateLocalPos0Attr().Set(local_pos0)
                joint.CreateLocalPos1Attr().Set(local_pos1)

                # Set local rotations (identity quaternions)
                joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

                # Configure joint drives for Y and Z rotation
                for axis in ["rotY", "rotZ"]:
                    limitAPI = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
                    limitAPI.CreateLowAttr(-110)
                    limitAPI.CreateHighAttr(110)

                    driveAPI = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
                    driveAPI.CreateTypeAttr("force")
                    driveAPI.CreateDampingAttr(damping)
                    driveAPI.CreateStiffnessAttr(stiffness)

        # Create target markers
        self.goal_markers = []
        marker_prim_path = self.cfg.goal_object_cfg.prim_path
        for seg_ind in range(num_segments):
            self.cfg.goal_object_cfg.prim_path = marker_prim_path + str(seg_ind)
            self.goal_markers.append(VisualizationMarkers(cfg=self.cfg.goal_object_cfg))

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # Translate the pusher based on the actions
        pusher_translation = self.actions[:, :2] * self.cfg.pusher_translation_scale
        cur_root = self.pusher.data.root_pos_w.clone()
        local_pos = cur_root[:, :3].clone() - self.scene.env_origins
        local_pos[:, :2] += pusher_translation
        min_xy = -0.5
        max_xy = 0.5
        local_pos[:, :2] = local_pos[:, :2].clamp(min_xy, max_xy)
        self.pusher_new_state = self.pusher.data.default_root_state.clone()
        self.pusher_new_state[:, :3] = local_pos + self.scene.env_origins


    def _apply_action(self):
        self.pusher.write_root_pose_to_sim(self.pusher_new_state[:, :7])

    def _get_observations(self) -> dict:
        # Flatten the rope segment positions and target positions
        rope_segments_pos = torch.cat(self.rope_segments_pos, dim=-1).view(self.num_envs, self.cfg.num_segments, 2)  # Current positions of rope segments
        rope_target_pos = self.rope_target_pos.view(self.num_envs, self.cfg.num_segments, 2)  # Target positions of rope segments

        # Compute relative positions between the pusher and each rope segment
        pusher_pos_expanded = self.pusher_pos[:, :2].unsqueeze(1).expand(-1, self.cfg.num_segments, -1)
        relative_positions = (rope_segments_pos - pusher_pos_expanded).view(self.num_envs, -1)

        # Include rope velocities in the observation
        rope_velocities = torch.stack([segment.data.root_vel_w for segment in self.rope_rigid_bodies], dim=-1)[:, :2, :].view(self.num_envs, -1)

        obs = torch.cat(
            (
                self.pusher_pos.view(self.num_envs, -1),  # Pusher position (3)
                rope_segments_pos.view(self.num_envs, -1),  # Rope segment positions (2 * num_segments)
                rope_target_pos.view(self.num_envs, -1),  # Target positions for rope segments (2 * num_segments)
                relative_positions.view(self.num_envs, -1),  # Relative positions between pusher and rope segments (2 * num_segments)
                rope_velocities.view(self.num_envs, -1),  # Rope velocities (2 * num_segments)
            ),
            dim=-1,
        )

        return {"policy": obs.to(self.device)}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
    
        # Compute euclidean distance between each segment of the rope and the respective target position
        rope_segments_pos = torch.cat(self.rope_segments_pos, dim=-1).view(self.num_envs, self.cfg.num_segments, 2)
        rope_target_pos = self.rope_target_pos.view(self.num_envs, self.cfg.num_segments, 2)

        dist_to_target = torch.sqrt(torch.sum((rope_segments_pos - rope_target_pos) ** 2, dim=-1)).to(self.device)

        # Compute reward based on proximity to the target position
        proximity_reward = -dist_to_target
        proximity_reward = torch.mean(proximity_reward, dim=1)

        # Temperature for proximity reward
        temp_proximity = 0.05

        proximity_reward_exp = torch.exp(proximity_reward / temp_proximity)

        # Compute the total reward
        reward = proximity_reward_exp

        # Log reward terms for debugging
        self.extras["log"] = {
            'proximity_reward_exp': proximity_reward_exp,
        }

        return reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # End the episode if the rope is close to the target or max steps reached
        self._compute_intermediate_values()
        rope_segments_pos_tensor = torch.cat(self.rope_segments_pos, dim=-1)
        distance_to_target = torch.norm(rope_segments_pos_tensor - self.rope_target_pos.view(self.num_envs, -1), dim=-1)
        success = distance_to_target < 0.01
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] |  torch.Tensor | None):
        # if env_ids is None:
        #     env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset table
        table_default_state = self.table.data.default_root_state.clone()[env_ids]
        table_default_state[:, :3] += self.scene.env_origins[env_ids]
        self.table.write_root_pose_to_sim(table_default_state[:, :7], env_ids)
        self.table.write_root_velocity_to_sim(table_default_state[:, 7:], env_ids)

        # reset pusher
        pusher_default_state = self.pusher.data.default_root_state.clone()[env_ids]
        pusher_default_state[:, :3] += self.scene.env_origins[env_ids]
        self.pusher.write_root_pose_to_sim(pusher_default_state[:, :7], env_ids)
        self.pusher.write_root_velocity_to_sim(pusher_default_state[:, 7:], env_ids)

        # reset rope
        for seg_ind, rope_segment in enumerate(self.rope_rigid_bodies):
            rope_default_state = rope_segment.data.default_root_state.clone()[env_ids]
            rope_default_state[:, :3] += self.scene.env_origins[env_ids]
            if seg_ind == 0:
                previous_pos = rope_default_state[:, :3].clone()
                rope_default_state[:, 0] += sample_uniform(-0.05, 0.05, (len(env_ids),), self.device)
                rope_default_state[:, 1] += sample_uniform(-0.05, 0.05, (len(env_ids),), self.device)
            else:
                angle = sample_uniform(-np.pi/6, np.pi/6, (len(env_ids),), self.device)
                rope_default_state[:, 0] = previous_pos[:, 0] + self.cfg.segment_length * torch.cos(angle)
                rope_default_state[:, 1] = previous_pos[:, 1] + self.cfg.segment_length * torch.sin(angle)
                previous_pos = rope_default_state[:, :3].clone()
            rope_segment.write_root_pose_to_sim(rope_default_state[:, :7], env_ids)
            rope_segment.write_root_velocity_to_sim(rope_default_state[:, 7:], env_ids)
            rope_segment.reset()

        # reset target pose
        rope_target_pos_all_env = torch.zeros((self.num_envs, self.cfg.num_segments, 3), dtype=torch.float, device=self.device)
        rope_target_pos_all_env[:, :, :2] = self.rope_target_pos.clone()
        rope_target_pos_all_env[:, :, 2] = self.first_segment_target_pos[2]
        rope_target_pos_3d = torch.zeros((len(env_ids), self.cfg.num_segments, 3), dtype=torch.float, device=self.device)

        rope_target_pos_3d = self.I_shape(env_ids)

        rope_target_pos_all_env[env_ids] = rope_target_pos_3d
        self.rope_target_pos = rope_target_pos_all_env[:, :, :2].clone()
        for seg_ind, marker in enumerate(self.goal_markers):
            marker.visualize(rope_target_pos_all_env[:, seg_ind, :] + self.scene.env_origins, self.goal_rot)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)
    
    
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        self.rope_segments_pos = [segment.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2] for segment in self.rope_rigid_bodies]
        self.pusher_pos = self.pusher.data.root_pos_w - self.scene.env_origins

    def I_shape(self, env_ids):
        rope_target_pos = torch.zeros((self.num_envs, self.cfg.num_segments, 3), dtype=torch.float, device=self.device)
        rope_target_pos[:, :, 0] = torch.linspace(self.first_segment_target_pos[0], self.first_segment_target_pos[0] + self.cfg.segment_length * (self.cfg.num_segments - 1), self.cfg.num_segments, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        rope_target_pos[:, :, 1] = self.first_segment_target_pos[1]
        rope_target_pos[:, :, 2] = self.first_segment_target_pos[2]

        return rope_target_pos[env_ids]