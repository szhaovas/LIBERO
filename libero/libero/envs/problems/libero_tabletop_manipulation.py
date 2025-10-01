from itertools import chain, combinations

import numpy as np
import robosuite.utils.transform_utils as T
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.robots import *
from libero.libero.envs.utils import rectangle2xyrange
from robosuite.utils.mjcf_utils import new_site
import mujoco


@register_problem
class Libero_Tabletop_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "main_table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})

        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/libero_tabletop_base_style.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
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

            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
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
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
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
                for (name, body) in query_dict.items():
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
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
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
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
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
            camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
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



@register_problem
class Libero_Spatial_Attack(Libero_Tabletop_Manipulation):
    table_bounds = [0.3, 0.38]
    next_to_bound = 0.15
    def __init__(self, bddl_file_name, *args, params, repair_env=False, repair_config=None, **kwargs):
        super().__init__(bddl_file_name, *args, **kwargs)

        # params have to be processed at the end of reset() or they will get 
        # overwritten 
        self._params = params

        if repair_env:
            assert repair_config is not None
            from docplex.mp.model import Context, Model
            context = Context.make_default_context()
            context.cplex_parameters.threads = 1
            context.cplex_parameters.dettimelimit = repair_config['time_limit']
            context.cplex_parameters.randomseed = repair_config['seed']
            context.cplex_parameters.optimalitytarget = 3
            self._mdl = Model(context=context)
            self._repair_config = repair_config
            
    @property
    def params(self):
        '''For now params should be an array listing object coordinates in the 
        following order:
          [
              akita_black_bowl_1_x, akita_black_bowl_1_y,
              akita_black_bowl_2_x, akita_black_bowl_2_y,
              cookies_1_x, cookies_1_y,
              glazed_rim_porcelain_ramekin_1_x,
              glazed_rim_porcelain_ramekin_1_y,
              plate_1_x, plate_1_y,
              light_x, light_y, light_z
              camera_x, camera_y, camera_z,
              table_r, table_g, table_b
          ]
        where akita_black_bowl_1_x, akita_black_bowl_1_y are relative to the
        ramekin's position (i.e. glazed_rim_porcelain_ramekin_1_x/y), and the
        remaining coordinates are relative to the table's position (i.e. same
        convention as in the original libero_spatial task suite).
        '''
        return self._params
    
    def _check_valid_placement(self):
        """Checks that the environment is valid. Right now it just checks that 
        no object overlap and all objects are within the table bounds.

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
                <= this_obj.horizontal_radius + other_obj.horizontal_radius
            ) and (
                this_z - other_z
                <= other_obj.top_offset[-1] - this_obj.bottom_offset[-1]
            ):
                raise ValueError(
                    "Overlapping objects:\n"
                    f"\t {this_obj.name} at {[this_x, this_y, this_z]}\n"
                    f"\t {other_obj.name} at {[other_x, other_y, other_z]}"
                )

        # Check everything is within the table bounds
        for movable_obj in self.objects_dict.values():
            obj_xy = self.sim.data.body_xpos[
                self.obj_body_id[movable_obj.name]
            ][:2]
            if np.any(
                (np.abs(obj_xy) + movable_obj.horizontal_radius) > self.table_bounds
            ):
                raise ValueError(
                    f"{movable_obj.name} at {obj_xy} outside of table bounds "
                    f"+-{self.table_bounds}"
                )
            
    def _construct_problem(self, mdl):
        """Constructs a CPLEX problem whose optimum corresponds to when all 
        movable objects are the closest to their starting locations while being 
        beyond a distance threshold to each other and all fixtures.

        To force objects to be at least some distance apart, we add constraints 
        forcing Linf distance >= horizontal_radius. We use Linf instead of L2 
        distance here because CPLEX only allows convex constraints, and L2 is 
        lower-bounded by Linf.
        """
        costs = []
        for this_obj, other_obj in combinations(
            chain(self.objects_dict.values(), self.fixtures_dict.values()), 2
        ):
            this_x, this_y, _ = self.sim.data.body_xpos[
                self.obj_body_id[this_obj.name]
            ]
            this_movable = this_obj.name in self.objects_dict
            other_x, other_y, _ = self.sim.data.body_xpos[
                self.obj_body_id[other_obj.name]
            ]
            other_movable = other_obj.name in self.objects_dict
    
            if this_movable:
                this_x_var = mdl.get_var_by_name(f"{this_obj.name}_x")
                this_y_var = mdl.get_var_by_name(f"{this_obj.name}_y")
                if this_x_var is None:
                    this_x_var = mdl.continuous_var(
                        name=f"{this_obj.name}_x",
                        lb=-self.table_bounds[0]+this_obj.horizontal_radius,
                        ub=self.table_bounds[0]-this_obj.horizontal_radius
                    )
                    this_y_var = mdl.continuous_var(
                        name=f"{this_obj.name}_y",
                        lb=-self.table_bounds[1]+this_obj.horizontal_radius,
                        ub=self.table_bounds[1]-this_obj.horizontal_radius
                    )
                    # minimize distance from starting location
                    costs.append((this_x_var-this_x)**2+(this_y_var-this_y)**2)
            else:
                this_x_var, this_y_var = this_x, this_y
            
            if other_movable:
                other_x_var = mdl.get_var_by_name(f"{other_obj.name}_x")
                other_y_var = mdl.get_var_by_name(f"{other_obj.name}_y")
                if other_x_var is None:
                    other_x_var = mdl.continuous_var(
                        name=f"{other_obj.name}_x",
                        lb=-self.table_bounds[0]+other_obj.horizontal_radius,
                        ub=self.table_bounds[0]-other_obj.horizontal_radius
                    )
                    other_y_var = mdl.continuous_var(
                        name=f"{other_obj.name}_y",
                        lb=-self.table_bounds[1]+other_obj.horizontal_radius,
                        ub=self.table_bounds[1]-other_obj.horizontal_radius
                    )
                    # minimize distance from starting location
                    costs.append((other_x_var-other_x)**2+(other_y_var-other_y)**2)
            else:
                other_x_var, other_y_var = other_x, other_y

            if this_movable or other_movable:
                # force objects to be some distance apart
                threshold = this_obj.horizontal_radius + \
                    other_obj.horizontal_radius + 1e-6
                mdl.add_constraint(
                    mdl.max(
                        mdl.abs(this_x_var - other_x_var), 
                        mdl.abs(this_y_var - other_y_var)
                    ) >= threshold
                )

        # Make sure plate_1 is close to ramekin, not plate_2
        plate_1_x_var = mdl.get_var_by_name("akita_black_bowl_1_x")
        plate_1_y_var = mdl.get_var_by_name("akita_black_bowl_1_y")
        ramekin_x_var = mdl.get_var_by_name("glazed_rim_porcelain_ramekin_1_x")
        ramekin_y_var = mdl.get_var_by_name("glazed_rim_porcelain_ramekin_1_y")
        plate_2_x_var = mdl.get_var_by_name("akita_black_bowl_2_x")
        plate_2_y_var = mdl.get_var_by_name("akita_black_bowl_2_y")
        mdl.add_constraint(
            mdl.max(
                mdl.abs(plate_1_x_var - ramekin_x_var), 
                mdl.abs(plate_1_y_var - ramekin_y_var)
            ) <= self.next_to_bound
        )
        mdl.add_constraint(
            mdl.max(
                mdl.abs(plate_2_x_var - ramekin_x_var), 
                mdl.abs(plate_2_y_var - ramekin_y_var)
            ) >= self.next_to_bound + 1e-6
        )
        
        mdl.minimize(sum(costs))

    def _place_objects(self, params):
        for idx, movable_obj in enumerate(self.objects_dict.values()):
            # Only update the objects' xy coordinates
            # TODO: Set akita_black_bowl_1_x/y relative to ramekin's coordinates
            start_i, _ = self.sim.data.model.get_joint_qpos_addr(
                movable_obj.joints[-1]
            )
            self.sim.data.qpos[start_i : start_i + 2] = params[
                2 * idx : 2 * idx + 2
            ]

        self.sim.forward()

    def reset(self):
        """Essentially the same reset as in robosuite MujocoEnv except it 
        modifies the environment according to :attr:`params` at the end.

        Returns:
            observations (dict): Same as MujocoEnv reset.
        """
        super().reset()

        # A bunch of modifications inspired from robosuite/utils/mjmod.py
        # Set lighting position
        self.sim.model.light_pos[self.sim.model.light_name2id('light1')] = self.params[10:13]

        # Set camera position
        self.sim.model.cam_pos[self.sim.model.camera_name2id('agentview')] = self.params[13:16]
        # TODO: Also allow changing camera rotation
        # (Maybe borrow the look_at function from simplerenv and change cam_pos and where the camera aims)

        # Set material color hint
        self.sim.model.mat_rgba[mujoco.mj_name2id(self.sim.model._model, int(mujoco.mjtObj.mjOBJ_MATERIAL), "table_texture")][:-1] = np.clip(self.params[16:19], 0, 1)
        # TODO: More objects
        # TODO: Add MILP repair for color (might be overkill...)
        
        # Set object arrangement
        self._place_objects(self.params[:10])
        try:
            self._check_valid_placement()
        except ValueError as e:
            if hasattr(self, '_mdl'):
                print(f'Repairing params {self.params[:10]}')
                self._construct_problem(self._mdl)
                repaired_params = self._mdl.solve()
                repaired_params = np.array(list(chain.from_iterable((
                        repaired_params.get_value(f'{movable_obj.name}_x'), repaired_params.get_value(f'{movable_obj.name}_y')) 
                        for movable_obj in self.objects_dict.values()
                    )))
                if repaired_params is None:
                    print('No repair solution found')
                    raise e
                print(f'New params: {repaired_params}')
                self._place_objects(repaired_params)
                self._check_valid_placement()
                self._params[:10] = repaired_params
            else:
                raise e

        observations = (
            self.viewer._get_observations(force_update=True)
            if self.viewer_get_obs
            else self._get_observations(force_update=True)
        )

        return observations

            
    def compute_spread_similarity(self):
        """Computes object clustering measures.

        Returns:
            Spread: A float between [0, 1]. The mean pairwise distance to the
                nearest neighbor normalized by the maximum possible pairwise
                distance within table bounds.
            Similarity: A float between [0, 1]. The average pairwise distance
                normalized by the maximum possible pairwise distance within
                table bounds. This is subtracted from 1 so that a higher value
                means more similar.
        """
        max_dist = np.linalg.norm(2*self.table_bounds)

        pairwise_dists = np.zeros((len(self.objects_dict), len(self.objects_dict)))
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