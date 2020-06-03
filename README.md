# tensorflow2.0-motion-planning
Implementation of simple motion planning task and deep reinforcement learning with tensorflow 2.x.

The goal of our agent is to find the best path(fast and safe). For reinforcement learning algorithm, I used TD3 and SAC for training.

This repository contains 3 environments : 2-DOF, 3-DOF, 3-DOF with two arms. And those have very few difference with each other. 

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


## Training Result
![3rr](https://user-images.githubusercontent.com/26384442/83638872-ca193e80-a5e4-11ea-9669-9ce07386179f.JPG)
#### td3 her for 3-DOF

<br/>

<br/>
 
![3srr](https://user-images.githubusercontent.com/26384442/83638961-f339cf00-a5e4-11ea-8e8b-302d78c9b087.JPG)
#### sac her for 3-DOF



