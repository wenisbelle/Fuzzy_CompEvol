# Description

This repository is part of the project that aims to create an efficient autonomous surveillance system that will provide constant delivery of service during the whole activity, using UAVs, for now, but having the possibility to extended to different ground agents.

For now, this repository contains the simulations in GradySim for the task allocation optimization, finding the new best position to verify after an encounter of after the current task is achieved. 

In this simulations it is being compared a fuzzy and a fitness function for the task allocation, both tuned through genetic algorithms. Both the trainning and the testing are presenting in this repository. 

## Instalation

### System Requirements
- Docker Engine
- Nvidia Container Toolkit 

The Nvidia container toolkit is not really required for the simulation to work, but the docker-compose file is configured to use it. So if you don't want to use GPU just modify the docker-compose.yaml file to remove this necessity. 

### Creating the Container

Fist, clone this project:

    git clone https://github.com/wenisbelle/Fuzzy_CompEvol.git

Build the docker image:

    docker build -t fuzzy_ga .

Then create the container with the configurations present in the .yaml file:

    docker compose up -d 

Always remember to give the screen permissions to the container before launching it:

    xhost local+

To interact with the container run:

    docker exec -it fuzzy_ga_container bash 

## Architecture


# TODO:
[ ] Improve the speed of the fuzzy look up table. 

[ ] Optimize not just the next target but also with which speed it should go to that point

[ ] Develop an energy consumption model, integrating with GradySim, improving the system from the just optimize distance traveled. 

[ ] Increase number of sets to be tunned in the fuzzy.

[ ] Make it possible to tune also the rules of the system. 

[ ] Make it suitable for MARL