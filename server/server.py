import asyncio
import json
import time
from dataclasses import dataclass

import numpy as np
import stable_baselines3 as sb3


@dataclass
class State:
    report: dict = None

    obs: np.ndarray = np.zeros(12, dtype=np.int32)
    phase: int = 1
    phase_start_tp: float = 0.0


# TODO: refactor me! Pack everything into a Server class
host, port = 'localhost', 8888
MSG_SIZE = 1024
HEADER_SIZE = 32
PHASE_LEN = 1 # seconds

state = State()

with open('ids.json') as f:
    ids = json.load(f)
cclock_to_obs = ids['cclock_to_obs']
start_lane_ids = ids['start_lane_ids']

phases = [1, 2]
policy = sb3.PPO.load('ppo_demo', device='cpu')


def extract_obs(report):
    obs = np.zeros(len(start_lane_ids), dtype=np.int32)
    for car in report['cars']:
        cclock_idx = car['lane']
        obs_idx = cclock_to_obs[cclock_idx]
        if obs_idx != -1:
            obs[obs_idx] += 1

    obs = obs.astype(np.float32)
    # obs = np.log2(1 + obs).astype(np.int32)
    # normally, these are bucketed logarithmically
    # but in our simple demo we have very few cars
    # so this bucketing strat doesn't work well in this case
    obs = obs / obs.max() * 4
    return np.clip(obs.astype(np.int32), 0, 5)


async def handle_connection(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f'New connection: {addr!r}')

    data = await reader.read(HEADER_SIZE)
    data = data.rstrip(b'\0')
    header = data.decode('ascii')

    if header == 'feed':
        await handle_feed(reader, writer)
    elif header == 'client':
        await handle_client(reader, writer)

    print(f'{addr!r} disconnected')
    writer.close()
    await writer.wait_closed()


async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f'New client: {addr!r}')
    message = json.dumps(state.report)
    writer.write(message.encode('ascii').ljust(MSG_SIZE, b'\0'))
    await writer.drain()

    print('Close the connection')
    writer.close()
    await writer.wait_closed()


async def handle_feed(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f'New camera feed: {addr!r}')

    while True:
        data = await reader.read(MSG_SIZE)
        if not data:
            break
        print('here')

        data = data.rstrip(b'\0')
        report = json.loads(data.decode('ascii'))
        print(f'Received feed from {addr!r}')

        obs = extract_obs(report)
        print('obs:', obs)

        state.report = report
        state.obs = obs

        now = time.time()
        if now - state.phase_start_tp > PHASE_LEN:
            action, _ = policy.predict(obs, deterministic=True)
            phase = phases[action]
            print('new phase:', phase)

            state.phase_start_tp = now
            state.phase = phase

        state.report['phase'] = state.phase
        print(json.dumps(report, indent=2))


async def main():
    server = await asyncio.start_server(
        handle_connection, host, port)

    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f'Serving on {addrs}')

    async with server:
        await server.serve_forever()

asyncio.run(main())
