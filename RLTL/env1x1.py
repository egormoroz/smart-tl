import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding

import cityflow
import numpy as np

import json
import random


class CityFlow1x1(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, cfgs, steps_per_episode=100):
        # self.cfgs = glob.glob('data/rl/cfg_*')
        self.cfgs = cfgs
        with open(cfgs[0]) as f:
            cfg = json.load(f)

        self.cf_engine = cityflow.Engine(cfgs[0], thread_num=1)
        # self.cf_engine.set_save_replay(False)

        with open(cfg['dir'] + cfg['flowFile']) as f:
            flow_cfg = json.load(f)

        self.all_lade_ids = [lane_id for lane_id in 
                             self.cf_engine.get_lane_vehicles()]

        start_road_ids = []
        for v in flow_cfg:
            start_road_ids.append(v['route'][0])

        self.start_lane_ids = []
        for lane_id in self.all_lade_ids:
            for road_id in start_road_ids:
                if lane_id.startswith(road_id):
                    self.start_lane_ids.append(lane_id)
                    break

        self.step_interval = float(cfg['interval'])

        # actual phases
        # self.phase_durs = np.array([10] * 9, dtype=np.int64)
        # self.phases = np.arange(9)

        # demo phases
        self.phase_durs = [30, 30]
        self.phases = [1, 2]

        # fuck it, let's hardcode the intersection id and number of TL states
        self.inter_id = 'intersection_1_1'
        # self.action_space = spaces.Discrete(9) # 9 TL phases
        self.action_space = spaces.Discrete(len(self.phases))


        n_start_lanes = len(self.start_lane_ids)
        max_vehicles_per_lane = 100
        n_buckets = int(np.log2(1 + max_vehicles_per_lane))

        self.observation_space = spaces.MultiDiscrete([n_buckets] * n_start_lanes)

        self.steps_per_episode = steps_per_episode
        self.current_step = 0
        self.is_done = False
        self.reward_range = (float('-inf'), 0)

        self.waiting_dict = {}
        self.last_tick_time = 0


    def step(self, action):
        assert self.action_space.contains(action), f'invalid action specified: {action}'

        cum_reward = 0
        for _ in range(self.phase_durs[action]):
            self.cf_engine.set_tl_phase(self.inter_id, self.phases[action])
            self.cf_engine.next_step()
            state, reward = self._get_state_reward()
            cum_reward += reward

        self.current_step += 1

        if self.is_done:
            logger.warn("You are calling 'step()' even though this environment "
                        "has already returned done = True. You should always call "
                        "'reset()' once you receive 'done = True' -- any further "
                        "steps are undefined behavior.")
            cum_reward = 0

        if self.current_step == self.steps_per_episode:
            self.is_done = True

        return state, cum_reward, self.is_done, {}

    def seed(self, n):
        self.cf_engine.set_random_seed(n)

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
            random.seed(seed)

        self.cf_engine = cityflow.Engine(random.choice(self.cfgs),
                                         thread_num=1)
        # self.cf_engine.set_save_replay(False)

        self.cf_engine.reset()
        self.is_done = False
        self.current_step = 0
        self.last_tick_time = 0
        self.waiting_dict.clear()

        state, _ = self._get_state_reward()

        return state

    def set_save_replay(self, save_replay):
        self.cf_engine.set_save_replay(save_replay)

    def set_replay_path(self, path):
        self.cf_engine.set_replay_file(path)

    def _get_state_reward(self):
        waiting_per_lane = self.cf_engine.get_lane_waiting_vehicle_count()

        n_start_lanes = len(self.start_lane_ids)
        counts = np.zeros(n_start_lanes, dtype=np.int64)
        for i in range(n_start_lanes):
            counts[i] = waiting_per_lane[self.start_lane_ids[i]]

        state = np.log2(1 + counts).astype(np.int64)

        cur_time = self.cf_engine.get_current_time()
        elapsed = cur_time - self.last_tick_time
        self.last_tick_time = cur_time
        impatience = 0.0
        alpha = 10**(1.0/100)
        for v_id, speed in self.cf_engine.get_vehicle_speed().items():
            if abs(speed) < 0.1:
                t = self.waiting_dict.get(v_id, 1) * elapsed * alpha
                self.waiting_dict[v_id] = t
                impatience += t

        reward = -counts.sum()# - min(impatience, 1e3)

        return state, reward

