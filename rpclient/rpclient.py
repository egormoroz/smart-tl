import asyncio
from picamera2 import Picamera2
from libcamera import controls
import time
from detect import detect
import json
import cv2 as cv

'''
NOTE: To test the client without Pi camera connected do the following:
1. Comment out the camera imports
2. Comment out the camera setup
3. Provide your frame in get_frame function
'''


host, port = 'localhost', 8888
MSG_SIZE = 1024
HEADER_SIZE = 32
HEADER = 'feed'.encode('ascii').ljust(HEADER_SIZE, b'\0')


def make_report(frame):
    result = detect(frame)
    if result is None:
        return None, 0
    cars, w, h = result

    car_info = []
    for (cx, cy), ratio, lane_idx in cars:
        car_info.append({
            'x': int(cx),
            'y': int(cy),
            'ratio': round(ratio, 2),
            'lane': int(lane_idx),
        })

    report = {
        'cars': car_info,
        'canvas': { 'w': w, 'h': h }
    }
    print(report)

    return json.dumps(report), len(car_info)


def get_frame(cam):
    # return cv.imread('smol.jpg')
    img = cam.capture_array()
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)


async def tcp_echo_client():
    reader, writer = await asyncio.open_connection(host, port)
    cam = setup_camera()
    # cam = None

    print('Sending header...')
    writer.write(HEADER)
    await writer.drain()

    while not writer.is_closing():
        message, n_cars = make_report(get_frame(cam))

        if message is None or len(message) > 1024:# or n_cars > 13:
            print('image processing failed, skipping this frame...')
            await asyncio.sleep(1)
        else:
            print('Sending report...')
            writer.write(message.encode('ascii').ljust(MSG_SIZE, b'\0'))
            await writer.drain()
            await asyncio.sleep(5)

    print('Close the connection')
    writer.close()
    await writer.wait_closed()


def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()


    return picam2


if __name__ == '__main__':
    asyncio.run(tcp_echo_client())
    
