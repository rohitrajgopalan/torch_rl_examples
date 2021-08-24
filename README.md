# Examples using TorchRL on OpenAI Gym Environments

This repository demonstrates how we can apply the TorchRL library to solve some basic OpenAI Gym environments. The intent is to explore solidatory Deep RL agents against those Deep RL agents that get assistance from a user-specified heuristic function for choosing actions. 

Before running these examples, <a href="https://github.com/rohitrajgopalan/torch_rl">download the TorchRL repository</> and do the following:
1. Navigate to the root of the torch_rl directory on your local machine on the terminal<br />
2. Run python setup.py sdist <br />
3. Run python -m pip install dist/torch-rl-10.0.tar.gz <br />
  
The environments I have used here are Cartpole, Blackjack, Lunar Lander (continuous and discrete), Mountain Car (continuous and discrete), Bipedal Walker and Gym-CCC's MultiRotor (which I have made my own copy for it to work with the TorchRL library). 
