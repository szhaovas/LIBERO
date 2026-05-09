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
    table_bounds = [-0.6, -0.61, 0.45, 0.59]
    repair_retry_limit = 5

    def __init__(
        self, bddl_file_name, *args, env_params=None, repair_config=None, **kwargs
    ):
        super().__init__(bddl_file_name, *args, **kwargs)

        # env_params have to be processed at the end of reset() or they will get
        # overwritten
        if env_params is not None:
            self._env_params = env_params.copy()

        self._repair_config = repair_config
        if self._repair_config is not None:
            import docplex.mp.model

            self._docplex_mp_model = docplex.mp.model

    @property
    def env_params(self):
        """For now env_params should be an array listing object coordinates in the
        following order:
          [
              akita_black_bowl_1_x, akita_black_bowl_1_y,
              akita_black_bowl_2_x, akita_black_bowl_2_y,
              cookies_1_x, cookies_1_y,
              glazed_rim_porcelain_ramekin_1_x,
              glazed_rim_porcelain_ramekin_1_y,
              plate_1_x, plate_1_y,
              light_x, light_y, light_z,
              camera_x, camera_y, camera_z,
              table_r, table_g, table_b,
              camera_r1, camera_r2, camera_r3,
              light_spec_r, light_spec_g, light_spec_b,
          ]
        """
        assert hasattr(self, "_env_params")
        return self._env_params

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

    def _check_valid_env_basic(self):
        """Checks that no object overlap and all objects are within the table
        bounds.

        Raises:
            ValueError
        """
        # Check the movable objects do not overlap
        for this_obj, other_obj in combinations(
            chain(self.objects_dict.values(), self.fixtures_dict.values()), 2
        ):
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
                    # this_obj is on top
                    clearance = this_z - other_z
                    min_clearance = abs(this_obj.bottom_offset[-1]) + abs(
                        other_obj.top_offset[-1]
                    )
                else:
                    # other_obj is on top
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

        # Check everything is within the table bounds
        for movable_obj in self.objects_dict.values():
            obj_xy = self.sim.data.body_xpos[
                self.obj_body_id[movable_obj.name]
            ][:2]
            if np.any(
                obj_xy - movable_obj.horizontal_radius
                < [self.table_bounds[0], self.table_bounds[1]]
            ) or np.any(
                obj_xy + movable_obj.horizontal_radius
                > [self.table_bounds[2], self.table_bounds[3]]
            ):
                raise ValueError(
                    f"{movable_obj.name} at {obj_xy} outside of table bounds "
                    f"{self.table_bounds}"
                )

    def _check_valid_env_task(self):
        raise NotImplementedError

    def _milp_build_basic_problem(self):
        """Add constraints to :attr:`_mdl` and costs to :attr:`_mdl_costs` to
        build a MILP problem whose optimum corresponds to when all movable
        objects are the closest to their starting locations while satisfying:
            1. Objects don't overlap with one another
            2. No object is on the plate at the start

        To force objects to be at least some distance apart, we add constraints
        forcing Linf distance >= horizontal_radius. We use Linf instead of L2
        distance here because CPLEX only allows convex constraints, and L2 is
        lower-bounded by Linf.
        """
        for this_obj, other_obj in combinations(
            chain(self.objects_dict.values(), self.fixtures_dict.values()), 2
        ):
            this_x, this_y, this_z = self.sim.data.body_xpos[
                self.obj_body_id[this_obj.name]
            ]
            this_movable = this_obj.name in self.objects_dict
            other_x, other_y, other_z = self.sim.data.body_xpos[
                self.obj_body_id[other_obj.name]
            ]
            other_movable = other_obj.name in self.objects_dict

            if this_movable:
                this_x_var = self._mdl.get_var_by_name(f"{this_obj.name}_x")
                this_y_var = self._mdl.get_var_by_name(f"{this_obj.name}_y")
                if this_x_var is None:
                    this_x_var = self._mdl.continuous_var(
                        name=f"{this_obj.name}_x",
                        lb=self.table_bounds[0]
                        + this_obj.horizontal_radius
                        + 1e-6,
                        ub=self.table_bounds[2]
                        - this_obj.horizontal_radius
                        - 1e-6,
                    )
                    this_y_var = self._mdl.continuous_var(
                        name=f"{this_obj.name}_y",
                        lb=self.table_bounds[1]
                        + this_obj.horizontal_radius
                        + 1e-6,
                        ub=self.table_bounds[3]
                        - this_obj.horizontal_radius
                        - 1e-6,
                    )
                    # minimize distance from starting location
                    self._mdl_costs.append(
                        (this_x_var - this_x) ** 2 + (this_y_var - this_y) ** 2
                    )
            else:
                this_x_var, this_y_var = this_x, this_y

            if other_movable:
                other_x_var = self._mdl.get_var_by_name(f"{other_obj.name}_x")
                other_y_var = self._mdl.get_var_by_name(f"{other_obj.name}_y")
                if other_x_var is None:
                    other_x_var = self._mdl.continuous_var(
                        name=f"{other_obj.name}_x",
                        lb=self.table_bounds[0]
                        + other_obj.horizontal_radius
                        + 1e-6,
                        ub=self.table_bounds[2]
                        - other_obj.horizontal_radius
                        - 1e-6,
                    )
                    other_y_var = self._mdl.continuous_var(
                        name=f"{other_obj.name}_y",
                        lb=self.table_bounds[1]
                        + other_obj.horizontal_radius
                        + 1e-6,
                        ub=self.table_bounds[3]
                        - other_obj.horizontal_radius
                        - 1e-6,
                    )
                    # minimize distance from starting location
                    self._mdl_costs.append(
                        (other_x_var - other_x) ** 2
                        + (other_y_var - other_y) ** 2
                    )
            else:
                other_x_var, other_y_var = other_x, other_y

            # Force objects to be some Linf distance apart on the xy plane
            # if they overlap vertically or if one of them is plate_1
            if this_z >= other_z:
                # this_obj is on top
                clearance = this_z - other_z
                min_clearance = abs(this_obj.bottom_offset[-1]) + abs(
                    other_obj.top_offset[-1]
                )
            else:
                # other_obj is on top
                clearance = other_z - this_z
                min_clearance = abs(this_obj.top_offset[-1]) + abs(
                    other_obj.bottom_offset[-1]
                )

            if (this_movable or other_movable) and (
                (clearance < min_clearance - 1e-6)
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

    def _milp_build_task_problem(self):
        """Adds task-specific constraints to :attr:`_mdl` and/or costs to
        :attr:`_mdl_costs`.

        For example, for task:
        "Pick the akita black bowl next to the plate and place it on the plate"

        Constraints need to be added to make sure only the first bowl is close
        to the ramekin.
        """
        raise NotImplementedError

    def _place_objects(self, env_params):
        for idx, movable_obj in enumerate(self.objects_dict.values()):
            # Only update the objects' xy coordinates
            start_i, _ = self.sim.data.model.get_joint_qpos_addr(
                movable_obj.joints[-1]
            )
            self.sim.data.qpos[start_i : start_i + 2] = env_params[
                2 * idx : 2 * idx + 2
            ]

        self.sim.forward()

    def check_valid_env(self):
        self._check_valid_env_basic()

    def milp_build_problem(self):
        assert hasattr(self, "_mdl")
        self._milp_build_basic_problem()

    def milp_solve_problem(self):
        assert hasattr(self, "_mdl")
        self._mdl.minimize(sum(self._mdl_costs))
        repaired_params = self._mdl.solve()

        if repaired_params is not None:
            repaired_params = np.array(
                list(
                    chain.from_iterable(
                        (
                            repaired_params.get_value(f"{movable_obj.name}_x"),
                            repaired_params.get_value(f"{movable_obj.name}_y"),
                        )
                        for movable_obj in self.objects_dict.values()
                    )
                )
            )

        return repaired_params

    def try_milp_repair(self):
        """Attemps to repair :attr:`env_params` for :attr:`repair_retry_limit`
        times. Set :attr:`env_params` to repaired env_params if repair was successful.
        Otherwise raise ValueError.
        """

        print(f"Repairing env_params: {self.env_params}")

        num_milp_retry = 0
        while num_milp_retry < self.repair_retry_limit:
            print(f"\t Attempt {num_milp_retry}/{self.repair_retry_limit}")

            self._reset_milp()
            self.milp_build_problem()
            repaired_params = self.milp_solve_problem()

            if repaired_params is None:
                print("\t Failed to find a solution...")
                num_milp_retry += 1
                continue
            else:
                self._place_objects(repaired_params)
                try:
                    self.check_valid_env()
                except Exception as e:
                    print(f"\t {e}")
                    num_milp_retry += 1
                    continue

                self._env_params[:2*len(self.objects_dict)] = repaired_params
                print(f"Found new env_params: {self.env_params}")
                return

        raise ValueError(
            f"Failed to repair env_params after {self.repair_retry_limit} attempts..."
        )

    def reset(self):
        """Essentially the same reset as in robosuite MujocoEnv except it
        modifies the environment according to :attr:`env_params` at the end.

        Returns:
            observations (dict): Same as MujocoEnv reset.
        """
        Libero_Tabletop_Manipulation.reset(self)

        if hasattr(self, "_env_params"):
            obj_xy_end_idx = 2 * len(self.objects_dict)

            # A bunch of modifications inspired from robosuite/utils/mjmod.py
            # Set lighting position
            self.sim.model.light_pos[self.sim.model.light_name2id("light1")] = (
                self.env_params[obj_xy_end_idx:obj_xy_end_idx+3]
            )

            # Set camera position
            self.sim.model.cam_pos[self.sim.model.camera_name2id("agentview")] = (
                self.env_params[obj_xy_end_idx+3:obj_xy_end_idx+6]
            )

            # Set material color hint
            self.sim.model.mat_rgba[
                mujoco.mj_name2id(
                    self.sim.model._model,
                    int(mujoco.mjtObj.mjOBJ_MATERIAL),
                    "table_texture",
                )
            ][:-1] = np.clip(self.env_params[obj_xy_end_idx+6:obj_xy_end_idx+9], 0, 1)
            # TODO: More objects

            # Set camera quaternion
            camera_R = R.from_quat(
                self.sim.model.cam_quat[self.sim.model.camera_name2id("agentview")]
            )
            R_noise = R.from_rotvec(self.env_params[obj_xy_end_idx+9:obj_xy_end_idx+12])
            camera_R *= R_noise
            self.sim.model.cam_quat[self.sim.model.camera_name2id("agentview")] = (
                camera_R.as_quat()
            )

            # Set lighting specular
            self.sim.model.light_specular[
                self.sim.model.light_name2id("light1")
            ] = np.clip(self.env_params[obj_xy_end_idx+12:obj_xy_end_idx+15], 0, 1)

            # Set object arrangement
            self._place_objects(self.env_params[:obj_xy_end_idx])

            try:
                self.check_valid_env()
            except Exception as e:
                if self._repair_config is not None:
                    print(e)
                    try:
                        self.try_milp_repair()
                    except:
                        raise
                else:
                    raise

        observations = (
            self.viewer._get_observations(force_update=True)
            if self.viewer_get_obs
            else self._get_observations(force_update=True)
        )

        return observations

    def compute_spread_similarity(self):
        """Computes object clustering measures.

        Returns:
            spread (float): A float between [0, 1]. The mean pairwise distance
                to the nearest neighbor normalized by the maximum possible
                pairwise distance within table bounds.
            similarity (float): A float between [0, 1]. The average pairwise
                distance normalized by the maximum possible pairwise distance
                within table bounds. This is subtracted from 1 so that a higher
                value means more similar.
        """
        max_dist = np.linalg.norm(
            [
                self.table_bounds[2] - self.table_bounds[0],  # x range
                self.table_bounds[3] - self.table_bounds[1],  # y range
            ]
        )

        pairwise_dists = np.zeros(
            (len(self.objects_dict), len(self.objects_dict))
        )
        for i, this_obj in enumerate(self.objects_dict.values()):
            this_xy = self.sim.data.body_xpos[self.obj_body_id[this_obj.name]][
                :1
            ]
            for j, other_obj in enumerate(self.objects_dict.values()):
                other_xy = self.sim.data.body_xpos[
                    self.obj_body_id[other_obj.name]
                ][:2]
                pairwise_dists[i, j] = (
                    np.linalg.norm(this_xy - other_xy) / max_dist
                )

        mean_dist = np.mean(pairwise_dists)

        # Don't count distance to self when taking min
        np.fill_diagonal(pairwise_dists, np.inf)
        min_dists = np.min(pairwise_dists, axis=1)

        return np.mean(min_dists), 1 - mean_dist


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

    def _check_valid_env_basic(self):
        """Overrides :meth:`Libero_Spatial_Attack._check_valid_env_basic`."""
        # Check the movable objects do not overlap
        for this_obj, other_obj in combinations(
            chain(self.objects_dict.values(), self.fixtures_dict.values()), 2
        ):
            # disable overlap check between akita_black_bowl_1 and
            # wooden_cabinet_1
            if this_obj.name in [
                "akita_black_bowl_1",
                "wooden_cabinet_1",
            ] and other_obj.name in ["akita_black_bowl_1", "wooden_cabinet_1"]:
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
                    # this_obj is on top
                    clearance = this_z - other_z
                    min_clearance = abs(this_obj.bottom_offset[-1]) + abs(
                        other_obj.top_offset[-1]
                    )
                else:
                    # other_obj is on top
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

    def _milp_build_basic_problem(self):
        """Overrides :meth:`Libero_Spatial_Attack._milp_build_basic_problem`."""
        for this_obj, other_obj in combinations(
            chain(self.objects_dict.values(), self.fixtures_dict.values()), 2
        ):
            this_x, this_y, this_z = self.sim.data.body_xpos[
                self.obj_body_id[this_obj.name]
            ]
            this_movable = this_obj.name in self.objects_dict
            other_x, other_y, other_z = self.sim.data.body_xpos[
                self.obj_body_id[other_obj.name]
            ]
            other_movable = other_obj.name in self.objects_dict

            if this_movable:
                this_x_var = self._mdl.get_var_by_name(f"{this_obj.name}_x")
                this_y_var = self._mdl.get_var_by_name(f"{this_obj.name}_y")
                if this_x_var is None:
                    this_x_var = self._mdl.continuous_var(
                        name=f"{this_obj.name}_x",
                        lb=self.table_bounds[0]
                        + this_obj.horizontal_radius
                        + 1e-6,
                        ub=self.table_bounds[2]
                        - this_obj.horizontal_radius
                        - 1e-6,
                    )
                    this_y_var = self._mdl.continuous_var(
                        name=f"{this_obj.name}_y",
                        lb=self.table_bounds[1]
                        + this_obj.horizontal_radius
                        + 1e-6,
                        ub=self.table_bounds[3]
                        - this_obj.horizontal_radius
                        - 1e-6,
                    )
                    # minimize distance from starting location
                    self._mdl_costs.append(
                        (this_x_var - this_x) ** 2 + (this_y_var - this_y) ** 2
                    )
            else:
                this_x_var, this_y_var = this_x, this_y

            if other_movable:
                other_x_var = self._mdl.get_var_by_name(f"{other_obj.name}_x")
                other_y_var = self._mdl.get_var_by_name(f"{other_obj.name}_y")
                if other_x_var is None:
                    other_x_var = self._mdl.continuous_var(
                        name=f"{other_obj.name}_x",
                        lb=self.table_bounds[0]
                        + other_obj.horizontal_radius
                        + 1e-6,
                        ub=self.table_bounds[2]
                        - other_obj.horizontal_radius
                        - 1e-6,
                    )
                    other_y_var = self._mdl.continuous_var(
                        name=f"{other_obj.name}_y",
                        lb=self.table_bounds[1]
                        + other_obj.horizontal_radius
                        + 1e-6,
                        ub=self.table_bounds[3]
                        - other_obj.horizontal_radius
                        - 1e-6,
                    )
                    # minimize distance from starting location
                    self._mdl_costs.append(
                        (other_x_var - other_x) ** 2
                        + (other_y_var - other_y) ** 2
                    )
            else:
                other_x_var, other_y_var = other_x, other_y

            # disable distance constraint between akita_black_bowl_1 and
            # wooden_cabinet_1
            if this_obj.name in [
                "akita_black_bowl_1",
                "wooden_cabinet_1",
            ] and other_obj.name in ["akita_black_bowl_1", "wooden_cabinet_1"]:
                continue

            # Force objects to be some Linf distance apart on the xy plane
            # if they overlap vertically
            if this_z >= other_z:
                # this_obj is on top
                clearance = this_z - other_z
                min_clearance = abs(this_obj.bottom_offset[-1]) + abs(
                    other_obj.top_offset[-1]
                )
            else:
                # other_obj is on top
                clearance = other_z - this_z
                min_clearance = abs(this_obj.top_offset[-1]) + abs(
                    other_obj.bottom_offset[-1]
                )

            if (this_movable or other_movable) and (
                (clearance < min_clearance - 1e-6)
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
        top_layer_x, top_layer_y, _ = self.sim.data.get_site_xpos(
            "wooden_cabinet_1_top_region"
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
        stove_x, stove_y, _ = self.sim.data.get_site_xpos(
            "flat_stove_1_cook_region"
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
        cabinet_x, cabinet_y, _ = self.sim.data.body_xpos[
            self.obj_body_id["wooden_cabinet_1"]
        ]
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
            pos=[0.76, -0.45, 1.83],
            quat=[
                0.81,
                0.35,
                0.18,
                0.42,
            ],
            camera_attribs={"fovy": "58"} # from RealSense D435i
        )

    def _load_model(self):
        super()._load_model()
        # Rotating the robot here since set_base_ori gets ignored in init and reset
        self.robots[0].robot_model.set_base_ori((0, 0, np.pi))


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
