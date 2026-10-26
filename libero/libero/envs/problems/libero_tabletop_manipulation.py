from itertools import chain, combinations

import mujoco
import numpy as np

np.set_printoptions(precision=2)
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import new_site
from scipy.spatial.transform import Rotation as R

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.robots import *
from libero.libero.envs.utils import rectangle2xyrange


@register_problem
class Libero_Tabletop_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "main_table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = kwargs["table_full_size"]
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        if "workspace_offset" not in kwargs:
            kwargs.update({"workspace_offset": (0, 0, 0.90)})
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {
                "robots": [
                    f"Mounted{robot_name}" for robot_name in kwargs["robots"]
                ]
            }
        )
        kwargs.update({"arena_type": "table"})

        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update(
                {"scene_xml": "scenes/libero_tabletop_base_style.xml"}
            )
        if (
            "scene_properties" not in kwargs
            or kwargs["scene_properties"] is None
        ):
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "light-gray",
                        "wall_style": "light-gray-plaster",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "table":
                continue

            for fixture_instance in self.parsed_problem["fixtures"][
                fixture_category
            ]:
                self.fixtures_dict[fixture_instance] = get_object_fn(
                    fixture_category
                )(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):

            if "main_table" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = (
                    (ranges[2] - ranges[0]) / 2,
                    (ranges[3] - ranges[1]) / 2,
                )
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2,
                    (ranges[3] + ranges[1]) / 2,
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone

                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for name, body in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = (
                                    SiteObject(
                                        name=site_name,
                                        parent_name=body.name,
                                        joints=[
                                            joint.get("name")
                                            for joint in joints
                                        ],
                                        size=site.get("size"),
                                        rgba=site.get("rgba"),
                                        site_type=site.get("type"),
                                        site_pos=site.get("pos"),
                                        site_quat=site.get("quat"),
                                        object_properties=body.object_properties,
                                    )
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()

    def _check_success(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        return result

    def _eval_predicate(self, state):
        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name)
                .object_properties["vis_site_names"]
                .items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if (
                    (self.sim.model.site_rgba[vis_g_id][3] <= 0)
                    and site_visible
                ) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0)
                    and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.6586131746834771, 0.0, 1.6103500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        # For visualization purpose
        mujoco_arena.set_camera(
            camera_name="frontview",
            pos=[1.0, 0.0, 1.48],
            quat=[0.56, 0.43, 0.43, 0.56],
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )


class Libero_Spatial_Attack(Libero_Tabletop_Manipulation):
    """Apply scene parameters and optionally repair invalid object placements.

    Args:
        bddl_file_name: Task definition file.
        *args: Positional arguments forwarded to the tabletop environment.
        env_params: Full ordered XY/scene/yaw/tint/gripper vector, or None for
            native sampling. Tint coordinates follow scene object order, excluding
            Akita black bowls whose stock black materials are retained.
        check_valid: Check physical and task constraints after applying parameters.
        repair_config: Optional CPLEX time_limit and seed for invalid placements.
        repair_mask: Optional D-element boolean mask in env_params order. False
            prevents repair from changing that coordinate's supplied value.
            Only object/fixture XY coordinates currently support repair.
        parameter_bounds: Optional D-element sequence of inclusive (lower, upper)
            pairs or None entries. XY repair obeys these bounds in addition to
            collision, task, and table constraints. None means no extra bounds.
        **kwargs: Keyword arguments forwarded to the tabletop environment.
    """

    table_bounds = [-0.6, -0.61, 0.45, 0.59]
    repair_retry_limit = 5
    scene_param_count = 15

    def __init__(
        self,
        bddl_file_name,
        *args,
        env_params=None,
        check_valid=True,
        repair_config=None,
        repair_mask=None,
        parameter_bounds=None,
        **kwargs,
    ):
        if env_params is not None:
            self._env_params = np.asarray(env_params, dtype=float).copy()
            self._skip_object_sampling = True
        elif repair_mask is not None or parameter_bounds is not None:
            raise ValueError(
                "Repair masks and bounds require explicit env_params"
            )

        self._check_valid = check_valid
        self._repair_config = repair_config

        super().__init__(bddl_file_name, *args, **kwargs)

        try:
            if env_params is not None:
                self._configure_repair(repair_mask, parameter_bounds)
        except (TypeError, ValueError):
            self.close()
            raise

        if self._repair_config is not None:
            import docplex.mp.model

            self._docplex_mp_model = docplex.mp.model

    @property
    def env_params(self):
        """Return the environment parameter vector.

        Coordinates for movable objects and fixtures come first, followed by
        scene appearance, object yaw and tint, and the initial gripper position:

          [
              akita_black_bowl_1_x, akita_black_bowl_1_y,
              akita_black_bowl_2_x, akita_black_bowl_2_y,
              cookies_1_x, cookies_1_y,
              glazed_rim_porcelain_ramekin_1_x,
              glazed_rim_porcelain_ramekin_1_y,
              plate_1_x, plate_1_y,
              wooden_cabinet_1_x, wooden_cabinet_1_y,
              flat_stove_1_x, flat_stove_1_y,
              light_x, light_y, light_z,
              camera_x, camera_y, camera_z,
              table_r, table_g, table_b,
              camera_r1, camera_r2, camera_r3,
              light_spec_r, light_spec_g, light_spec_b,
              akita_black_bowl_1_yaw, akita_black_bowl_2_yaw,
              cookies_1_yaw, glazed_rim_porcelain_ramekin_1_yaw,
              plate_1_yaw, wooden_cabinet_1_yaw, flat_stove_1_yaw,
              cookies_1_r, cookies_1_g, cookies_1_b,
              glazed_rim_porcelain_ramekin_1_r,
              glazed_rim_porcelain_ramekin_1_g,
              glazed_rim_porcelain_ramekin_1_b,
              plate_1_r, plate_1_g, plate_1_b,
              wooden_cabinet_1_r, wooden_cabinet_1_g,
              wooden_cabinet_1_b,
              flat_stove_1_r, flat_stove_1_g, flat_stove_1_b,
              gripper_x, gripper_y, gripper_z,
          ]
        """
        assert hasattr(self, "_env_params")
        return self._env_params

    def _tinted_objects(self):
        """Select tintable scene objects in the shared simulator vector order."""
        # Multiplying the stock black bowl RGB offers no useful tint variation.
        return [
            scene_obj
            for scene_obj in self._scene_objects()
            if scene_obj.category_name
            not in {"akita_black_bowl", "large_akita_black_bowl"}
        ]

    def _configure_repair(self, repair_mask, parameter_bounds):
        """Validate coordinate masks and bounds once, before any repair model."""
        count = self._env_param_slices()["end"]
        if self._env_params.shape != (count,) or not np.all(np.isfinite(self._env_params)):
            raise ValueError(f"Expected {count} finite env_params")
        mask = np.ones(count, dtype=bool) if repair_mask is None else np.asarray(repair_mask)
        if mask.shape != (count,) or mask.dtype.kind != "b":
            raise ValueError(f"repair_mask must contain {count} booleans")
        bounds = np.asarray(
            [(-np.inf, np.inf) if pair is None else pair for pair in parameter_bounds]
            if parameter_bounds is not None
            else [(-np.inf, np.inf)] * count,
            dtype=float,
        )
        if (
            bounds.shape != (count, 2)
            or np.any(np.isnan(bounds))
            or np.any(bounds[:, 0] > bounds[:, 1])
        ):
            raise ValueError(f"parameter_bounds must contain {count} ordered pairs or None entries")
        self._repair_mask = mask.copy()
        self._parameter_bounds = bounds.copy()

    def _apply_repair_constraints(self):
        """Keep disabled XY coordinates fixed and restrict every repair to its range."""
        for index, scene_obj in enumerate(self._scene_objects()):
            for offset, axis in enumerate("xy"):
                coordinate = 2 * index + offset
                variable = self._mdl.get_var_by_name(f"{scene_obj.name}_{axis}")
                if not self._repair_mask[coordinate]:
                    self._mdl.add_constraint(variable == self._env_params[coordinate])
                lower, upper = self._parameter_bounds[coordinate]
                if np.isfinite(lower):
                    self._mdl.add_constraint(variable >= lower)
                if np.isfinite(upper):
                    self._mdl.add_constraint(variable <= upper)

    def _env_param_slices(self):
        """Build slices for the canonical environment parameter layout."""
        scene_object_count = len(self.objects_dict) + len(self.fixtures_dict)
        position_xy_end = 2 * scene_object_count
        scene_end = position_xy_end + self.scene_param_count
        yaw_end = scene_end + scene_object_count
        tint_end = yaw_end + 3 * len(self._tinted_objects())
        return {
            "position_xy": slice(0, position_xy_end),
            "scene": slice(position_xy_end, scene_end),
            "yaw": slice(scene_end, yaw_end),
            "tint": slice(yaw_end, tint_end),
            "gripper_xyz": slice(tint_end, tint_end + 3),
            "end": tint_end + 3,
        }

    def _reset_milp(self):
        context = self._docplex_mp_model.Context.make_default_context()
        context.cplex_parameters.threads = 1
        context.cplex_parameters.dettimelimit = self._repair_config[
            "time_limit"
        ]
        context.cplex_parameters.randomseed = self._repair_config["seed"]
        context.cplex_parameters.optimalitytarget = 3
        self._mdl = self._docplex_mp_model.Model(context=context)
        self._mdl_costs = []

    def _scene_objects(self):
        """Return movable objects and fixtures in their parameter order."""
        return list(
            chain(self.objects_dict.values(), self.fixtures_dict.values())
        )

    def _ignored_overlap_pair(self, this_obj, other_obj):
        """Allow task subclasses to exempt an intentional object overlap."""
        return False

    def _descendant_body_ids(self, object_name):
        """Find the root and child MuJoCo bodies belonging to one object."""
        body_ids = {self.obj_body_id[object_name]}
        for body_id in range(self.sim.model.nbody):
            if self.sim.model.body_parentid[body_id] in body_ids:
                body_ids.add(body_id)
        return body_ids

    def _gripper_collision_object_names(self):
        """Return scene objects currently contacting a gripper collision geom."""
        gripper_geom_ids = {
            self.sim.model.geom_name2id(name)
            for name in self.robots[0].gripper.contact_geoms
        }
        geom_to_object = {}
        for scene_obj in self._scene_objects():
            body_ids = self._descendant_body_ids(scene_obj.name)
            for geom_id in np.flatnonzero(
                np.isin(self.sim.model.geom_bodyid, list(body_ids))
            ):
                geom_to_object[int(geom_id)] = scene_obj.name

        collisions = set()
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            geom_1, geom_2 = int(contact.geom1), int(contact.geom2)
            if geom_1 in gripper_geom_ids and geom_2 in geom_to_object:
                collisions.add(geom_to_object[geom_2])
            elif geom_2 in gripper_geom_ids and geom_1 in geom_to_object:
                collisions.add(geom_to_object[geom_1])
        return collisions

    def _check_valid_env_basic(self):
        """Check scene collisions, table bounds, and gripper feasibility."""
        scene_objects = self._scene_objects()

        for this_obj, other_obj in combinations(scene_objects, 2):
            if self._ignored_overlap_pair(this_obj, other_obj):
                continue

            this_x, this_y, this_z = self.sim.data.body_xpos[
                self.obj_body_id[this_obj.name]
            ]
            other_x, other_y, other_z = self.sim.data.body_xpos[
                self.obj_body_id[other_obj.name]
            ]
            if (
                np.linalg.norm((this_x - other_x, this_y - other_y))
                < this_obj.horizontal_radius + other_obj.horizontal_radius
            ):
                if this_z >= other_z:
                    clearance = this_z - other_z
                    min_clearance = abs(this_obj.bottom_offset[-1]) + abs(
                        other_obj.top_offset[-1]
                    )
                else:
                    clearance = other_z - this_z
                    min_clearance = abs(this_obj.top_offset[-1]) + abs(
                        other_obj.bottom_offset[-1]
                    )

                if clearance < min_clearance - 1e-6:
                    raise ValueError(
                        "Overlapping objects:\n"
                        f"\t {this_obj.name} at {np.array([this_x, this_y, this_z])}\n"
                        f"\t {other_obj.name} at {np.array([other_x, other_y, other_z])}"
                    )

        # A circular footprint is conservative for every yaw angle.
        for scene_obj in scene_objects:
            obj_xy = self.sim.data.body_xpos[
                self.obj_body_id[scene_obj.name]
            ][:2]
            if np.any(
                obj_xy - scene_obj.horizontal_radius
                < [self.table_bounds[0], self.table_bounds[1]]
            ) or np.any(
                obj_xy + scene_obj.horizontal_radius
                > [self.table_bounds[2], self.table_bounds[3]]
            ):
                raise ValueError(
                    f"{scene_obj.name} at {obj_xy} outside of table bounds "
                    f"{self.table_bounds}"
                )

        gripper_xyz = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        # Table XY bounds constrain placed objects, not the robot's workspace.
        # Reachable gripper poses may lie outside that box. IK checks reachability
        # when applying the pose; height and collisions are checked here.
        if gripper_xyz[2] < self.workspace_offset[2]:
            raise ValueError(
                f"Gripper at {gripper_xyz} starts below the table surface "
                f"z={self.workspace_offset[2]}"
            )
        collisions = self._gripper_collision_object_names()
        if collisions:
            raise ValueError(
                "Gripper starts in collision with "
                + ", ".join(sorted(collisions))
            )

    def _reset_internal(self):
        """Apply explicit positions after native fixture and support-height setup."""
        super()._reset_internal()
        if hasattr(self, "_env_params"):
            self._place_objects(
                self._env_params[self._env_param_slices()["position_xy"]]
            )

    def _check_valid_env_task(self):
        raise NotImplementedError

    def _milp_xy(self, scene_obj):
        """Return or create the repair variables for an object's XY position."""
        x, y, _ = self.sim.data.body_xpos[self.obj_body_id[scene_obj.name]]
        x_var = self._mdl.get_var_by_name(f"{scene_obj.name}_x")
        y_var = self._mdl.get_var_by_name(f"{scene_obj.name}_y")
        if x_var is None:
            x_var = self._mdl.continuous_var(
                name=f"{scene_obj.name}_x",
                lb=self.table_bounds[0]
                + scene_obj.horizontal_radius
                + 1e-6,
                ub=self.table_bounds[2]
                - scene_obj.horizontal_radius
                - 1e-6,
            )
            y_var = self._mdl.continuous_var(
                name=f"{scene_obj.name}_y",
                lb=self.table_bounds[1]
                + scene_obj.horizontal_radius
                + 1e-6,
                ub=self.table_bounds[3]
                - scene_obj.horizontal_radius
                - 1e-6,
            )
            self._mdl_costs.append((x_var - x) ** 2 + (y_var - y) ** 2)
        return x_var, y_var

    def _milp_site_xy(self, site_name, object_name):
        """Express a child site's XY pose relative to its repairable object."""
        scene_obj = self.get_object(object_name)
        object_x, object_y, _ = self.sim.data.body_xpos[
            self.obj_body_id[object_name]
        ]
        site_x, site_y, _ = self.sim.data.get_site_xpos(site_name)
        object_x_expr, object_y_expr = self._milp_xy(scene_obj)
        return (
            object_x_expr + site_x - object_x,
            object_y_expr + site_y - object_y,
        )

    def _milp_build_basic_problem(self):
        """Build table-bound and collision constraints for repairable objects."""
        scene_objects = self._scene_objects()

        # Create variables even when an object has no active pairwise constraint.
        for scene_obj in scene_objects:
            self._milp_xy(scene_obj)

        for this_obj, other_obj in combinations(scene_objects, 2):
            if self._ignored_overlap_pair(this_obj, other_obj):
                continue

            _, _, this_z = self.sim.data.body_xpos[
                self.obj_body_id[this_obj.name]
            ]
            _, _, other_z = self.sim.data.body_xpos[
                self.obj_body_id[other_obj.name]
            ]
            this_x_var, this_y_var = self._milp_xy(this_obj)
            other_x_var, other_y_var = self._milp_xy(other_obj)

            if this_z >= other_z:
                clearance = this_z - other_z
                min_clearance = abs(this_obj.bottom_offset[-1]) + abs(
                    other_obj.top_offset[-1]
                )
            else:
                clearance = other_z - this_z
                min_clearance = abs(this_obj.top_offset[-1]) + abs(
                    other_obj.bottom_offset[-1]
                )

            if (
                clearance < min_clearance - 1e-6
                or this_obj.name == "plate_1"
                or other_obj.name == "plate_1"
            ):
                threshold = (
                    this_obj.horizontal_radius
                    + other_obj.horizontal_radius
                    + 0.01
                )
                self._mdl.add_constraint(
                    self._mdl.max(
                        self._mdl.abs(this_x_var - other_x_var),
                        self._mdl.abs(this_y_var - other_y_var),
                    )
                    >= threshold
                )

        # If a supplied gripper pose is in contact, allow repair to move the
        # offending object away while keeping the requested gripper pose fixed.
        gripper_x, gripper_y, _ = self.sim.data.site_xpos[
            self.robots[0].eef_site_id
        ]
        for object_name in self._gripper_collision_object_names():
            scene_obj = self.get_object(object_name)
            obj_x_var, obj_y_var = self._milp_xy(scene_obj)
            self._mdl.add_constraint(
                self._mdl.max(
                    self._mdl.abs(obj_x_var - gripper_x),
                    self._mdl.abs(obj_y_var - gripper_y),
                )
                >= scene_obj.horizontal_radius + 0.04
            )

    def _milp_build_task_problem(self):
        """Add task-specific constraints to the current repair model."""
        raise NotImplementedError

    def _place_objects(self, position_params):
        """Apply XY positions, sampled support heights, and fixture positions."""
        for idx, scene_obj in enumerate(self._scene_objects()):
            xy = position_params[2 * idx : 2 * idx + 2]
            if scene_obj.name in self.objects_dict:
                start_i, _ = self.sim.data.model.get_joint_qpos_addr(
                    scene_obj.joints[-1]
                )
                self.sim.data.qpos[start_i : start_i + 2] = xy
                sampled_placement = getattr(
                    self, "_sampled_object_placements", {}
                ).get(scene_obj.name)
                self.sim.data.qpos[start_i + 2] = (
                    sampled_placement[0][2]
                    if sampled_placement is not None
                    else self.workspace_offset[2]
                    + self.z_offset
                    - scene_obj.bottom_offset[-1]
                )
            else:
                body_id = self.obj_body_id[scene_obj.name]
                self.sim.model.body_pos[body_id][:2] = xy
        self.sim.forward()

    def _set_object_yaws(self, normalized_yaws):
        """Map normalized yaws to [-pi, pi] and apply them about world Z."""
        normalized_yaws = np.asarray(normalized_yaws)
        if np.any((normalized_yaws < 0) | (normalized_yaws > 1)):
            raise ValueError("Object yaw parameters must be within [0, 1]")

        yaws = 2 * np.pi * normalized_yaws - np.pi
        for scene_obj, yaw in zip(self._scene_objects(), yaws):
            quat = np.array(
                [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            )
            if scene_obj.name in self.objects_dict:
                start_i, _ = self.sim.data.model.get_joint_qpos_addr(
                    scene_obj.joints[-1]
                )
                self.sim.data.qpos[start_i + 3 : start_i + 7] = quat
            else:
                self.sim.model.body_quat[
                    self.obj_body_id[scene_obj.name]
                ] = quat
        self.sim.forward()

    def _cache_object_tint_baselines(self):
        """Cache stock RGB values once per model to prevent compounded tints."""
        if getattr(self, "_tint_baseline_model", None) is self.sim.model:
            return

        self._tint_baseline_model = self.sim.model
        self._base_material_rgb = self.sim.model.mat_rgba[:, :3].copy()
        self._base_geom_rgb = self.sim.model.geom_rgba[:, :3].copy()

    def _set_object_tint(self, object_name, tint):
        """Multiply an object's stock visual RGB values by the requested tint."""
        self._cache_object_tint_baselines()
        body_ids = self._descendant_body_ids(object_name)
        geom_ids = np.flatnonzero(
            np.isin(self.sim.model.geom_bodyid, list(body_ids))
        )
        material_ids = set()
        for geom_id in geom_ids:
            material_id = int(self.sim.model.geom_matid[geom_id])
            if material_id >= 0:
                material_ids.add(material_id)
            elif self.sim.model.geom_group[geom_id] == 1:
                self.sim.model.geom_rgba[geom_id, :3] = (
                    self._base_geom_rgb[geom_id] * tint
                )
        for material_id in material_ids:
            self.sim.model.mat_rgba[material_id, :3] = (
                self._base_material_rgb[material_id] * tint
            )

    def _set_initial_gripper_position(self, target_position):
        """Use damped least-squares IK while retaining the stock orientation."""
        robot = self.robots[0]
        site_name = robot.gripper.important_sites["grip_site"]
        target_orientation = self.sim.data.get_site_xmat(site_name).reshape(3, 3).copy()
        qpos_indexes = np.asarray(robot._ref_joint_pos_indexes)
        qvel_indexes = np.asarray(robot._ref_joint_vel_indexes)
        joint_ranges = self.sim.model.jnt_range[robot._ref_joint_indexes]

        for _ in range(200):
            current_position = self.sim.data.get_site_xpos(site_name).copy()
            current_orientation = self.sim.data.get_site_xmat(site_name).reshape(3, 3)
            position_error = target_position - current_position
            orientation_error = R.from_matrix(
                target_orientation @ current_orientation.T
            ).as_rotvec()
            error = np.concatenate((position_error, orientation_error))
            if np.linalg.norm(position_error) < 1e-5 and np.linalg.norm(
                orientation_error
            ) < 1e-4:
                break

            jacobian = np.vstack(
                (
                    self.sim.data.get_site_jacp(site_name)[:, qvel_indexes],
                    self.sim.data.get_site_jacr(site_name)[:, qvel_indexes],
                )
            )
            damping = 1e-4
            joint_delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            delta_norm = np.linalg.norm(joint_delta)
            if delta_norm > 0.2:
                joint_delta *= 0.2 / delta_norm

            joint_positions = self.sim.data.qpos[qpos_indexes] + joint_delta
            limited = joint_ranges[:, 0] != joint_ranges[:, 1]
            joint_positions[limited] = np.clip(
                joint_positions[limited],
                joint_ranges[limited, 0] + 1e-6,
                joint_ranges[limited, 1] - 1e-6,
            )
            robot.set_robot_joint_positions(joint_positions)

        actual_position = self.sim.data.get_site_xpos(site_name).copy()
        if np.linalg.norm(actual_position - target_position) >= 2e-4:
            raise ValueError(
                f"Initial gripper position {target_position} is unreachable; "
                f"closest IK result was {actual_position}"
            )
        self.sim.data.qvel[qvel_indexes] = 0
        robot.controller.update_initial_joints(robot._joint_positions)

    def check_valid_env(self):
        self._check_valid_env_basic()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()

    def milp_solve_problem(self):
        assert hasattr(self, "_mdl")
        self._mdl.minimize(sum(self._mdl_costs))
        repaired_solution = self._mdl.solve()

        if repaired_solution is None:
            return None
        return np.array(
            list(
                chain.from_iterable(
                    (
                        repaired_solution.get_value(f"{scene_obj.name}_x"),
                        repaired_solution.get_value(f"{scene_obj.name}_y"),
                    )
                    for scene_obj in self._scene_objects()
                )
            )
        )

    def try_milp_repair(self):
        """Repair parameterized XY positions, or raise after repeated failure."""
        print(f"Repairing env_params: {self.env_params}")

        num_milp_retry = 0
        while num_milp_retry < self.repair_retry_limit:
            print(f"\t Attempt {num_milp_retry}/{self.repair_retry_limit}")

            self._reset_milp()
            self.milp_build_problem()
            self._apply_repair_constraints()
            repaired_params = self.milp_solve_problem()

            if repaired_params is None:
                print("\t Failed to find a solution...")
                num_milp_retry += 1
                continue

            self._place_objects(repaired_params)
            try:
                self.check_valid_env()
            except Exception as e:
                print(f"\t {e}")
                num_milp_retry += 1
                continue

            position_slice = self._env_param_slices()["position_xy"]
            self._env_params[position_slice] = repaired_params
            print(f"Found new env_params: {self.env_params}")
            return

        raise ValueError(
            f"Failed to repair env_params after {self.repair_retry_limit} attempts..."
        )

    def reset(self):
        """Reset, then apply the supplied scene and initial gripper parameters."""
        Libero_Tabletop_Manipulation.reset(self)

        if hasattr(self, "_env_params"):
            slices = self._env_param_slices()
            if len(self.env_params) != slices["end"]:
                raise ValueError(
                    f"Expected {slices['end']} env_params, got "
                    f"{len(self.env_params)}"
                )
            if not np.all(np.isfinite(self.env_params)):
                raise ValueError("env_params must contain only finite values")
            scene_params = self.env_params[slices["scene"]]

            self.sim.model.light_pos[self.sim.model.light_name2id("light1")] = (
                scene_params[:3]
            )
            self.sim.model.cam_pos[self.sim.model.camera_name2id("agentview")] = (
                scene_params[3:6]
            )
            self.sim.model.mat_rgba[
                mujoco.mj_name2id(
                    self.sim.model._model,
                    int(mujoco.mjtObj.mjOBJ_MATERIAL),
                    "table_texture",
                )
            ][:-1] = np.clip(scene_params[6:9], 0, 1)

            camera_rotation = R.from_quat(
                self.sim.model.cam_quat[
                    self.sim.model.camera_name2id("agentview")
                ]
            )
            camera_rotation *= R.from_rotvec(scene_params[9:12])
            self.sim.model.cam_quat[
                self.sim.model.camera_name2id("agentview")
            ] = camera_rotation.as_quat()

            self.sim.model.light_specular[
                self.sim.model.light_name2id("light1")
            ] = np.clip(scene_params[12:15], 0, 1)

            self._place_objects(self.env_params[slices["position_xy"]])
            self._set_object_yaws(self.env_params[slices["yaw"]])
            tint_params = self.env_params[slices["tint"]].reshape(-1, 3)
            for scene_obj, tint in zip(
                self._tinted_objects(), np.clip(tint_params, 0, 1)
            ):
                self._set_object_tint(scene_obj.name, tint)
            self._set_initial_gripper_position(
                self.env_params[slices["gripper_xyz"]]
            )

            if self._check_valid:
                try:
                    self.check_valid_env()
                except Exception as e:
                    if self._repair_config is not None:
                        print(e)
                        self.try_milp_repair()
                    else:
                        raise

        observations = (
            self.viewer._get_observations(force_update=True)
            if self.viewer_get_obs
            else self._get_observations(force_update=True)
        )
        return observations



@register_problem
class Task_0(Libero_Spatial_Attack):
    """This task requires that akita_black_bowl_1 is between plate_1 and
    glazed_rim_porcelain_ramekin_1 and akita_black_bowl_2 is not.

    To satisfy this requirement, we check that only akita_black_bowl_1 is close
    to the center point between plate_1 and glazed_rim_porcelain_ramekin_1.
    """

    between_tolerance = 0.3  # should be between 0 and 0.5
    not_between_tolerance = 0.5

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _check_valid_env_task(self):
        plate_x, plate_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["plate_1"]
        ]
        ramekin_x, ramekin_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["glazed_rim_porcelain_ramekin_1"]
        ]
        midpoint_x = (plate_x + ramekin_x) / 2
        midpoint_y = (plate_y + ramekin_y) / 2

        bowl_1_x, bowl_1_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if not (
            abs(bowl_1_x - midpoint_x)
            <= self.between_tolerance * abs(ramekin_x - plate_x)
            and abs(bowl_1_y - midpoint_y)
            <= self.between_tolerance * abs(ramekin_y - plate_y)
        ):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y])} is not between "
                f"plate_1 at {np.array([plate_x, plate_y])} and "
                f"glazed_rim_porcelain_ramekin_1 at {np.array([ramekin_x, ramekin_y])}"
            )

        bowl_2_x, bowl_2_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_2"]
        ]
        if abs(bowl_2_x - midpoint_x) < self.not_between_tolerance * abs(
            ramekin_x - plate_x
        ) and abs(bowl_2_y - midpoint_y) < self.not_between_tolerance * abs(
            ramekin_y - plate_y
        ):
            raise ValueError(
                f"akita_black_bowl_2 at {np.array([bowl_2_x, bowl_2_y])} is between "
                f"plate_1 at {np.array([plate_x, plate_y])} and "
                f"glazed_rim_porcelain_ramekin_1 at {np.array([ramekin_x, ramekin_y])}"
            )

    def _milp_build_task_problem(self):
        plate_x_var = self._mdl.get_var_by_name("plate_1_x")
        plate_y_var = self._mdl.get_var_by_name("plate_1_y")
        ramekin_x_var = self._mdl.get_var_by_name(
            "glazed_rim_porcelain_ramekin_1_x"
        )
        ramekin_y_var = self._mdl.get_var_by_name(
            "glazed_rim_porcelain_ramekin_1_y"
        )

        # Make sure there's enough space between plate_1 and
        # glazed_rim_porcelain_ramekin_1 to fit akita_black_bowl_1
        threshold = (
            self.objects_dict["plate_1"].horizontal_radius
            + self.objects_dict[
                "glazed_rim_porcelain_ramekin_1"
            ].horizontal_radius
            + 2 * self.objects_dict["akita_black_bowl_1"].horizontal_radius
        )
        self._mdl.add_constraint(
            self._mdl.max(
                self._mdl.abs(plate_x_var - ramekin_x_var),
                self._mdl.abs(plate_y_var - ramekin_y_var),
            )
            >= threshold + 1e-6
        )

        # Make sure akita_black_bowl_1 is between plate_1 and
        # glazed_rim_porcelain_ramekin_1, i.e. both its xy are close to the
        # midpoint between them
        midpoint_x_var = (plate_x_var + ramekin_x_var) / 2
        midpoint_y_var = (plate_y_var + ramekin_y_var) / 2
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        self._mdl.add_constraint(
            self._mdl.abs(bowl_1_x_var - midpoint_x_var)
            <= self.between_tolerance
            * self._mdl.abs(ramekin_x_var - plate_x_var)
            - 1e-6
        )
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        self._mdl.add_constraint(
            self._mdl.abs(bowl_1_y_var - midpoint_y_var)
            <= self.between_tolerance
            * self._mdl.abs(ramekin_y_var - plate_y_var)
            - 1e-6
        )

        # Make sure akita_black_bowl_2 is not between plate_1 and
        # glazed_rim_porcelain_ramekin_1, i.e. at least one of its xy isn't
        # close to the midpoint between them
        bowl_2_x_var = self._mdl.get_var_by_name("akita_black_bowl_2_x")
        bowl_2_x_awyf_mid = self._mdl.binary_var(name="bowl_2_x_awyf_mid")
        self._mdl.add_indicator(
            bowl_2_x_awyf_mid,
            self._mdl.abs(bowl_2_x_var - midpoint_x_var)
            >= self.not_between_tolerance
            * self._mdl.abs(ramekin_x_var - plate_x_var)
            + 1e-6,
        )
        bowl_2_y_var = self._mdl.get_var_by_name("akita_black_bowl_2_y")
        bowl_2_y_awyf_mid = self._mdl.binary_var(name="bowl_2_y_awyf_mid")
        self._mdl.add_indicator(
            bowl_2_y_awyf_mid,
            self._mdl.abs(bowl_2_y_var - midpoint_y_var)
            >= self.not_between_tolerance
            * self._mdl.abs(ramekin_y_var - plate_y_var)
            + 1e-6,
        )
        self._mdl.add_constraint(bowl_2_x_awyf_mid + bowl_2_y_awyf_mid >= 1)


@register_problem
class Task_1(Libero_Spatial_Attack):
    """This task requires that akita_black_bowl_1 is close to
    glazed_rim_porcelain_ramekin_1 and akita_black_bowl_2 is not
    """

    next_to_bound = 0.1
    far_from_bound = 0.2

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _check_valid_env_task(self):
        ramekin_x, ramekin_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["glazed_rim_porcelain_ramekin_1"]
        ]

        padding = (
            self.objects_dict[
                "glazed_rim_porcelain_ramekin_1"
            ].horizontal_radius
            + self.objects_dict["akita_black_bowl_1"].horizontal_radius
        )

        bowl_1_x, bowl_1_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if (
            max(abs(bowl_1_x - ramekin_x), abs(bowl_1_y - ramekin_y))
            > padding + self.next_to_bound
        ):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y])} is not close "
                f"to ramekin_1 at {np.array([ramekin_x, ramekin_y])}"
            )

        bowl_2_x, bowl_2_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_2"]
        ]
        if (
            max(abs(bowl_2_x - ramekin_x), abs(bowl_2_y - ramekin_y))
            < padding + self.far_from_bound
        ):
            raise ValueError(
                f"akita_black_bowl_2 at {np.array([bowl_2_x, bowl_2_y])} is close to "
                f"ramekin_1 at {np.array([ramekin_x, ramekin_y])}"
            )

    def _milp_build_task_problem(self):
        ramekin_x_var = self._mdl.get_var_by_name(
            "glazed_rim_porcelain_ramekin_1_x"
        )
        ramekin_y_var = self._mdl.get_var_by_name(
            "glazed_rim_porcelain_ramekin_1_y"
        )

        padding = (
            self.objects_dict[
                "glazed_rim_porcelain_ramekin_1"
            ].horizontal_radius
            + self.objects_dict["akita_black_bowl_1"].horizontal_radius
        )

        # make sure akita_black_bowl_1 is close to ramekin_1
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        self._mdl.add_constraint(
            self._mdl.max(
                self._mdl.abs(bowl_1_x_var - ramekin_x_var),
                self._mdl.abs(bowl_1_y_var - ramekin_y_var),
            )
            <= padding + self.next_to_bound - 1e-6
        )

        # make sure akita_black_bowl_2 is not close to ramekin_1
        bowl_2_x_var = self._mdl.get_var_by_name("akita_black_bowl_2_x")
        bowl_2_y_var = self._mdl.get_var_by_name("akita_black_bowl_2_y")
        self._mdl.add_constraint(
            self._mdl.max(
                self._mdl.abs(bowl_2_x_var - ramekin_x_var),
                self._mdl.abs(bowl_2_y_var - ramekin_y_var),
            )
            >= padding + self.far_from_bound + 1e-6
        )


@register_problem
class Task_2(Libero_Spatial_Attack):
    """This task requires that akita_black_bowl_1 is within bounds
    ``main_table_table_center`` and akita_black_bowl_2 is not.
    """

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _check_valid_env_task(self):
        xl, yl, xh, yh = self.parsed_problem["regions"][
            "main_table_table_center"
        ]["ranges"][0]

        bowl_1_x, bowl_1_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if not (xl <= bowl_1_x <= xh and yl <= bowl_1_y <= yh):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y])} is not within "
                f"bounds {xl}<=x<={xh}; {yl}<=y<={yh}"
            )

        bowl_2_x, bowl_2_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_2"]
        ]
        if xl < bowl_2_x < xh or yl < bowl_2_y < yh:
            raise ValueError(
                f"akita_black_bowl_2 at {np.array([bowl_2_x, bowl_2_y])} is within "
                f"bounds {xl}<=x<={xh}; {yl}<=y<={yh}"
            )

    def _milp_build_task_problem(self):
        xl, yl, xh, yh = self.parsed_problem["regions"][
            "main_table_table_center"
        ]["ranges"][0]

        # make sure akita_black_bowl_1 is within bounds
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        self._mdl.add_constraint(bowl_1_x_var >= xl + 1e-6)
        self._mdl.add_constraint(bowl_1_x_var <= xh - 1e-6)
        self._mdl.add_constraint(bowl_1_y_var >= yl + 1e-6)
        self._mdl.add_constraint(bowl_1_y_var <= yh - 1e-6)

        # make sure akita_black_bowl_2_x is not within x bounds, i.e. either
        # below lb or above ub
        bowl_2_x_var = self._mdl.get_var_by_name("akita_black_bowl_2_x")
        bowl_2_x_smt_lb = self._mdl.binary_var(name="bowl_2_x_smt_lb")
        self._mdl.add_indicator(bowl_2_x_smt_lb, bowl_2_x_var <= xl - 1e-6, 1)
        self._mdl.add_indicator(bowl_2_x_smt_lb, bowl_2_x_var >= xh + 1e-6, 0)
        # make sure akita_black_bowl_2_y is not within y bounds
        bowl_2_y_var = self._mdl.get_var_by_name("akita_black_bowl_2_y")
        bowl_2_y_smt_lb = self._mdl.binary_var(name="bowl_2_y_smt_lb")
        self._mdl.add_indicator(bowl_2_y_smt_lb, bowl_2_y_var <= yl - 1e-6, 1)
        self._mdl.add_indicator(bowl_2_y_smt_lb, bowl_2_y_var >= yh + 1e-6, 0)


@register_problem
class Task_3(Libero_Spatial_Attack):
    """This task requires that akita_black_bowl_1 is on cookies_1."""

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _check_valid_env_task(self):
        cookies_x, cookies_y, cookies_z = self.sim.data.body_xpos[
            self.obj_body_id["cookies_1"]
        ]
        bowl_1_x, bowl_1_y, bowl_1_z = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if not np.all(np.isclose([cookies_x, cookies_y], [bowl_1_x, bowl_1_y])):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y, bowl_1_z])} is not "
                f"on cookies_1 at {np.array([cookies_x, cookies_y, cookies_z])}"
            )

        # akita_black_bowl_2 cannot be on cookies_1 without overlapping with
        # akita_black_bowl_1, so no need to handle seperately

    def _milp_build_task_problem(self):
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        cookies_x_var = self._mdl.get_var_by_name("cookies_1_x")
        cookies_y_var = self._mdl.get_var_by_name("cookies_1_y")
        self._mdl.add_constraint(bowl_1_x_var == cookies_x_var)
        self._mdl.add_constraint(bowl_1_y_var == cookies_y_var)

        # akita_black_bowl_2 cannot be on cookies_1 without overlapping with
        # akita_black_bowl_1, so no need to handle seperately


@register_problem
class Task_4(Libero_Spatial_Attack):
    """This task requires that akita_black_bowl_1 is on
    ``wooden_cabinet_1_top_region``

    By design of this task, akita_black_bowl_1 has to overlap with
    wooden_cabinet_1, so the corresponding checks and MILP constraints
    between akita_black_bowl_1 and wooden_cabinet_1 are disabled.
    """

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _ignored_overlap_pair(self, this_obj, other_obj):
        """Ignore the bowl/cabinet overlap required by this task."""
        return {this_obj.name, other_obj.name} == {
            "akita_black_bowl_1",
            "wooden_cabinet_1",
        }

    def _check_valid_env_task(self):
        top_layer_x, top_layer_y, top_layer_z = self.sim.data.get_site_xpos(
            "wooden_cabinet_1_top_region"
        )
        bowl_1_x, bowl_1_y, bowl_1_z = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if not np.all(
            np.isclose([top_layer_x, top_layer_y], [bowl_1_x, bowl_1_y])
        ):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y, bowl_1_z])} is not "
                f"in the cabinet at {np.array([top_layer_x, top_layer_y, top_layer_z])}"
            )

        # akita_black_bowl_2 cannot be in wooden_cabinet_1_top_region without
        # overlapping with akita_black_bowl_1, so no need to handle seperately

    def _milp_build_task_problem(self):
        top_layer_x, top_layer_y = self._milp_site_xy(
            "wooden_cabinet_1_top_region", "wooden_cabinet_1"
        )
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        self._mdl.add_constraint(bowl_1_x_var == top_layer_x)
        self._mdl.add_constraint(bowl_1_y_var == top_layer_y)

        # akita_black_bowl_2 cannot be in wooden_cabinet_1_top_region without
        # overlapping with akita_black_bowl_1, so no need to handle seperately


@register_problem
class Task_5(Libero_Spatial_Attack):
    """This task requires that akita_black_bowl_1 is on
    glazed_rim_porcelain_ramekin_1.
    """

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _check_valid_env_task(self):
        ramekin_x, ramekin_y, ramekin_z = self.sim.data.body_xpos[
            self.obj_body_id["glazed_rim_porcelain_ramekin_1"]
        ]
        bowl_1_x, bowl_1_y, bowl_1_z = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if not np.all(np.isclose([ramekin_x, ramekin_y], [bowl_1_x, bowl_1_y])):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y, bowl_1_z])} is not "
                "on glazed_rim_porcelain_ramekin_1 at "
                f"{np.array([ramekin_x, ramekin_y, ramekin_z])}"
            )

        # akita_black_bowl_2 cannot be on glazed_rim_porcelain_ramekin_1 without
        # overlapping with akita_black_bowl_1, so no need to handle seperately

    def _milp_build_task_problem(self):
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        ramekin_x_var = self._mdl.get_var_by_name(
            "glazed_rim_porcelain_ramekin_1_x"
        )
        ramekin_y_var = self._mdl.get_var_by_name(
            "glazed_rim_porcelain_ramekin_1_y"
        )
        self._mdl.add_constraint(bowl_1_x_var == ramekin_x_var)
        self._mdl.add_constraint(bowl_1_y_var == ramekin_y_var)

        # akita_black_bowl_2 cannot be on glazed_rim_porcelain_ramekin_1 without
        # overlapping with akita_black_bowl_1, so no need to handle seperately


@register_problem
class Task_6(Libero_Spatial_Attack):
    """This task requires that akita_black_bowl_1 is close to cookies_1 and
    akita_black_bowl_2 is not
    """

    next_to_bound = 0.1
    far_from_bound = 0.2

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _check_valid_env_task(self):
        cookies_x, cookies_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["cookies_1"]
        ]

        padding = (
            self.objects_dict["cookies_1"].horizontal_radius
            + self.objects_dict["akita_black_bowl_1"].horizontal_radius
        )

        bowl_1_x, bowl_1_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if (
            max(abs(bowl_1_x - cookies_x), abs(bowl_1_y - cookies_y))
            > padding + self.next_to_bound
        ):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y])} is not close "
                f"to cookies_1 at {np.array([cookies_x, cookies_y])}"
            )

        bowl_2_x, bowl_2_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_2"]
        ]
        if (
            max(abs(bowl_2_x - cookies_x), abs(bowl_2_y - cookies_y))
            < padding + self.far_from_bound
        ):
            raise ValueError(
                f"akita_black_bowl_2 at {np.array([bowl_2_x, bowl_2_y])} is close to "
                f"cookies_1 at {np.array([cookies_x, cookies_y])}"
            )

    def _milp_build_task_problem(self):
        cookies_x_var = self._mdl.get_var_by_name("cookies_1_x")
        cookies_y_var = self._mdl.get_var_by_name("cookies_1_y")

        padding = (
            self.objects_dict["cookies_1"].horizontal_radius
            + self.objects_dict["akita_black_bowl_1"].horizontal_radius
        )

        # make sure akita_black_bowl_1 is close to cookies_1
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        self._mdl.add_constraint(
            self._mdl.max(
                self._mdl.abs(bowl_1_x_var - cookies_x_var),
                self._mdl.abs(bowl_1_y_var - cookies_y_var),
            )
            <= padding + self.next_to_bound - 1e-6
        )

        # make sure akita_black_bowl_2 is not close to cookies_1
        bowl_2_x_var = self._mdl.get_var_by_name("akita_black_bowl_2_x")
        bowl_2_y_var = self._mdl.get_var_by_name("akita_black_bowl_2_y")
        self._mdl.add_constraint(
            self._mdl.max(
                self._mdl.abs(bowl_2_x_var - cookies_x_var),
                self._mdl.abs(bowl_2_y_var - cookies_y_var),
            )
            >= padding + self.far_from_bound + 1e-6
        )


@register_problem
class Task_7(Libero_Spatial_Attack):
    """This task requires that akita_black_bowl_1 is on
    ``flat_stove_1_cook_region``
    """

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _check_valid_env_task(self):
        stove_x, stove_y, stove_z = self.sim.data.get_site_xpos(
            "flat_stove_1_cook_region"
        )
        bowl_1_x, bowl_1_y, bowl_1_z = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if not np.all(np.isclose([stove_x, stove_y], [bowl_1_x, bowl_1_y])):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y, bowl_1_z])} is not "
                f"on the stove at {np.array([stove_x, stove_y, stove_z])}"
            )

        # akita_black_bowl_2 cannot be in flat_stove_1_cook_region without
        # overlapping with akita_black_bowl_1, so no need to handle seperately

    def _milp_build_task_problem(self):
        stove_x, stove_y = self._milp_site_xy(
            "flat_stove_1_cook_region", "flat_stove_1"
        )
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        self._mdl.add_constraint(bowl_1_x_var == stove_x)
        self._mdl.add_constraint(bowl_1_y_var == stove_y)

        # akita_black_bowl_2 cannot be in flat_stove_1_cook_region without
        # overlapping with akita_black_bowl_1, so no need to handle seperately


@register_problem
class Task_8(Libero_Spatial_Attack):
    """This task requires that akita_black_bowl_1 is close to plate_1 and
    akita_black_bowl_2 is not
    """

    next_to_bound = 0.1
    far_from_bound = 0.2

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _check_valid_env_task(self):
        plate_x, plate_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["plate_1"]
        ]

        padding = (
            self.objects_dict["plate_1"].horizontal_radius
            + self.objects_dict["akita_black_bowl_1"].horizontal_radius
        )

        bowl_1_x, bowl_1_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if (
            max(abs(bowl_1_x - plate_x), abs(bowl_1_y - plate_y))
            > padding + self.next_to_bound
        ):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y])} is not close "
                f"to plate_1 at {np.array([plate_x, plate_y])}"
            )

        bowl_2_x, bowl_2_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_2"]
        ]
        if (
            max(abs(bowl_2_x - plate_x), abs(bowl_2_y - plate_y))
            < padding + self.far_from_bound
        ):
            raise ValueError(
                f"akita_black_bowl_2 at {np.array([bowl_2_x, bowl_2_y])} is close to "
                f"plate_1 at {np.array([plate_x, plate_y])}"
            )

    def _milp_build_task_problem(self):
        plate_x_var = self._mdl.get_var_by_name("plate_1_x")
        plate_y_var = self._mdl.get_var_by_name("plate_1_y")

        padding = (
            self.objects_dict["plate_1"].horizontal_radius
            + self.objects_dict["akita_black_bowl_1"].horizontal_radius
        )

        # make sure akita_black_bowl_1 is close to plate_1
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        self._mdl.add_constraint(
            self._mdl.max(
                self._mdl.abs(bowl_1_x_var - plate_x_var),
                self._mdl.abs(bowl_1_y_var - plate_y_var),
            )
            <= padding + self.next_to_bound - 1e-6
        )

        # make sure akita_black_bowl_2 is not close to plate_1
        bowl_2_x_var = self._mdl.get_var_by_name("akita_black_bowl_2_x")
        bowl_2_y_var = self._mdl.get_var_by_name("akita_black_bowl_2_y")
        self._mdl.add_constraint(
            self._mdl.max(
                self._mdl.abs(bowl_2_x_var - plate_x_var),
                self._mdl.abs(bowl_2_y_var - plate_y_var),
            )
            >= padding + self.far_from_bound + 1e-6
        )


@register_problem
class Task_9(Libero_Spatial_Attack):
    """This task requires that akita_black_bowl_1 is on
    ``wooden_cabinet_1_top_side``
    """

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _check_valid_env_task(self):
        cabinet_x, cabinet_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["wooden_cabinet_1"]
        ]
        bowl_1_x, bowl_1_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if not np.all(np.isclose([cabinet_x, cabinet_y], [bowl_1_x, bowl_1_y])):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y])} is not "
                f"on the cabinet at {np.array([cabinet_x, cabinet_y])}"
            )

        # akita_black_bowl_2 cannot be on the cabinet without overlapping with
        # akita_black_bowl_1, so no need to handle seperately

    def _milp_build_task_problem(self):
        cabinet_x, cabinet_y = self._milp_xy(
            self.fixtures_dict["wooden_cabinet_1"]
        )
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        self._mdl.add_constraint(bowl_1_x_var == cabinet_x)
        self._mdl.add_constraint(bowl_1_y_var == cabinet_y)

        # akita_black_bowl_2 cannot be on the cabinet without overlapping with
        # akita_black_bowl_1, so no need to handle seperately


class Jaco_Custom_Tabletop(Libero_Spatial_Attack):
    table_bounds = [-0.8, -0.4, 0.8, 0.4]

    def __init__(
        self, bddl_file_name, *args, env_params=None, repair_config=None, **kwargs
    ):
        kwargs.update({"robots": ["Jaco6DOF"]})
        kwargs.update({"scene_xml": "scenes/libero_tabletop_custom_style.xml"})
        kwargs.update({"scene_properties": {
                "floor_style": "light-gray",
                "wall_style": "white",
                "table_visual_half_size": (0.9, 0.8, 0.025),
                "autoset_wall_texture": False
            }
        })
        kwargs.update({"table_full_size": (1.60, 0.80, 0.05)})
        kwargs.update({"workspace_offset": (0, 0, 0.92)})

        super().__init__(bddl_file_name, *args, env_params=env_params, repair_config=repair_config, **kwargs)

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.76, 0.45, 1.83],
            quat=[
                0.42,
                0.18,
                0.35,
                0.82,
            ],
            camera_attribs={"fovy": "58"},  # from RealSense D435i
        )

    def _load_model(self):
        super()._load_model()
        # Rotating the robot here since set_base_ori gets ignored in init and reset
        self.robots[0].robot_model.set_base_ori((0, 0, 0))

    def _check_gripper_contact(self, object_model):
        gripper_geoms = [
            geom
            for geom_group in self.robots[0].gripper.important_geoms.values()
            for geom in geom_group
        ]
        return self.check_contact(gripper_geoms, object_model.contact_geoms)

    def _check_success(self):
        """Check whether the bowl clears its tabletop resting height by 0.1 m."""
        bowl = self.objects_dict["akita_black_bowl_1"]
        bowl_height = self.sim.data.body_xpos[self.obj_body_id[bowl.name]][2]
        # The larger bowl's center already exceeds table height + 0.1 at rest.
        bottom_height = bowl_height + bowl.bottom_offset[2]
        return bool(bottom_height >= self.workspace_offset[2] + 0.1)


@register_problem
class Jaco_Custom_Tabletop_Task_0(Jaco_Custom_Tabletop):
    """This task requires that akita_black_bowl_1 is within bounds
    ``main_table_table_center`` and akita_black_bowl_2 is not.
    """

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _check_valid_env_task(self):
        xl, yl, xh, yh = self.parsed_problem["regions"][
            "main_table_table_center"
        ]["ranges"][0]

        bowl_1_x, bowl_1_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if not (xl <= bowl_1_x <= xh and yl <= bowl_1_y <= yh):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y])} is not within "
                f"bounds {xl}<=x<={xh}; {yl}<=y<={yh}"
            )

        bowl_2_x, bowl_2_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_2"]
        ]
        if xl < bowl_2_x < xh or yl < bowl_2_y < yh:
            raise ValueError(
                f"akita_black_bowl_2 at {np.array([bowl_2_x, bowl_2_y])} is within "
                f"bounds {xl}<=x<={xh}; {yl}<=y<={yh}"
            )

    def _milp_build_task_problem(self):
        xl, yl, xh, yh = self.parsed_problem["regions"][
            "main_table_table_center"
        ]["ranges"][0]

        # make sure akita_black_bowl_1 is within bounds
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        self._mdl.add_constraint(bowl_1_x_var >= xl + 1e-6)
        self._mdl.add_constraint(bowl_1_x_var <= xh - 1e-6)
        self._mdl.add_constraint(bowl_1_y_var >= yl + 1e-6)
        self._mdl.add_constraint(bowl_1_y_var <= yh - 1e-6)

        # make sure akita_black_bowl_2_x is not within x bounds, i.e. either
        # below lb or above ub
        bowl_2_x_var = self._mdl.get_var_by_name("akita_black_bowl_2_x")
        bowl_2_x_smt_lb = self._mdl.binary_var(name="bowl_2_x_smt_lb")
        self._mdl.add_indicator(bowl_2_x_smt_lb, bowl_2_x_var <= xl - 1e-6, 1)
        self._mdl.add_indicator(bowl_2_x_smt_lb, bowl_2_x_var >= xh + 1e-6, 0)
        # make sure akita_black_bowl_2_y is not within y bounds
        bowl_2_y_var = self._mdl.get_var_by_name("akita_black_bowl_2_y")
        bowl_2_y_smt_lb = self._mdl.binary_var(name="bowl_2_y_smt_lb")
        self._mdl.add_indicator(bowl_2_y_smt_lb, bowl_2_y_var <= yl - 1e-6, 1)
        self._mdl.add_indicator(bowl_2_y_smt_lb, bowl_2_y_var >= yh + 1e-6, 0)


@register_problem
class Jaco_Custom_Tabletop_Task_1(Jaco_Custom_Tabletop):
    """This task requires that akita_black_bowl_1 is close to plate_1 and
    akita_black_bowl_2 is not
    """

    next_to_bound = 0.1
    far_from_bound = 0.2

    def check_valid_env(self):
        self._check_valid_env_basic()
        self._check_valid_env_task()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()
        self._milp_build_task_problem()

    def _check_valid_env_task(self):
        plate_x, plate_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["plate_1"]
        ]

        padding = (
            self.objects_dict["plate_1"].horizontal_radius
            + self.objects_dict["akita_black_bowl_1"].horizontal_radius
        )

        bowl_1_x, bowl_1_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_1"]
        ]
        if (
            max(abs(bowl_1_x - plate_x), abs(bowl_1_y - plate_y))
            > padding + self.next_to_bound
        ):
            raise ValueError(
                f"akita_black_bowl_1 at {np.array([bowl_1_x, bowl_1_y])} is not close "
                f"to plate_1 at {np.array([plate_x, plate_y])}"
            )

        bowl_2_x, bowl_2_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["akita_black_bowl_2"]
        ]
        if (
            max(abs(bowl_2_x - plate_x), abs(bowl_2_y - plate_y))
            < padding + self.far_from_bound
        ):
            raise ValueError(
                f"akita_black_bowl_2 at {np.array([bowl_2_x, bowl_2_y])} is close to "
                f"plate_1 at {np.array([plate_x, plate_y])}"
            )

    def _milp_build_task_problem(self):
        plate_x_var = self._mdl.get_var_by_name("plate_1_x")
        plate_y_var = self._mdl.get_var_by_name("plate_1_y")

        padding = (
            self.objects_dict["plate_1"].horizontal_radius
            + self.objects_dict["akita_black_bowl_1"].horizontal_radius
        )

        # make sure akita_black_bowl_1 is close to plate_1
        bowl_1_x_var = self._mdl.get_var_by_name("akita_black_bowl_1_x")
        bowl_1_y_var = self._mdl.get_var_by_name("akita_black_bowl_1_y")
        self._mdl.add_constraint(
            self._mdl.max(
                self._mdl.abs(bowl_1_x_var - plate_x_var),
                self._mdl.abs(bowl_1_y_var - plate_y_var),
            )
            <= padding + self.next_to_bound - 1e-6
        )

        # make sure akita_black_bowl_2 is not close to plate_1
        bowl_2_x_var = self._mdl.get_var_by_name("akita_black_bowl_2_x")
        bowl_2_y_var = self._mdl.get_var_by_name("akita_black_bowl_2_y")
        self._mdl.add_constraint(
            self._mdl.max(
                self._mdl.abs(bowl_2_x_var - plate_x_var),
                self._mdl.abs(bowl_2_y_var - plate_y_var),
            )
            >= padding + self.far_from_bound + 1e-6
        )
