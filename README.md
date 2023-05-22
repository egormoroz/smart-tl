# A toy real time traffic light control system

## Overview

This is a toy prototype of a traffic light control system. It takes photos
of a road junction, extracts vehicle positions and runs an RL agent
to predict the optimal traffic light phase. The training code can be 
found in RLTL folder.


There are 3 main components, each can be found in the respectively named folder
- Server. Runs RL agent, receives positional info and sends current 
state to webui client.
- Camera client. Executes on a Raspberry Pi controller with a camera connected 
to it. Takes photos, extracts positional info and sends it to server.
- Webui client. Recieves state info from server and displays the overview of the 
junction with the selected traffic light phase.


## Dependencies
- CityFlow
- OpenAI gym
- StableBaselines3 
- PyTorch
- OpenCV
- picamera2 and libcamera for the Raspberry client

