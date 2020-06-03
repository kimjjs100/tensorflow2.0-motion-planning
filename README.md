# tensorflow2.0-motion-planning
Implementation of simple motion planning task and deep reinforcement learning with tensorflow 2.x.

The goal of our agent is to find the best path(fast and safe). For reinforcement learning algorithm, I used TD3 and SAC for training.

This repository contains 3 environment : 2-DOF, 3-DOF, 3-DOF with two arms. And those have very few difference with each other. 

I hope it would be helpful for your study.

<br/> 

## Description
#### td3_her_path :

TD3(Twin Delayed DDPG) + HER(Hindsight experience replay) implementation for 2-DOF path planning.
 
#### sac_her_path :

Soft actor critic + HER implementation for 2-DOF path planning.&nbsp;

#### path_plan_dist :

TD3 for 3-DOF environment(And I try to use mirrored strategy but it doesn't works well..)

#### path_plan_sac :

sac for 3-DOF environment

#### path_plan_two :

td3 for 3-DOF environment with two manipulator

<br/>

## Requirements
tensorflow>=2.0.0

cpprb

numpy>=1.16 

<br/>


## Result

