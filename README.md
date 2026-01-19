# Turtlebot3 DQN Finetuning

## Content
### Outline : Whole Frames
### Algorithms
### Logics - Agent
### Logs


## Whole Frames

### Turtlebot3 DQN
Based on Turtlebot3 DQN

- DQN_Finetuning_Agent (dqn_agent)
- DQN_Finetuning_Gazebo (gazebo_interface)
- DQN_Finetuning_Environment (rl_environment)

After get 'Best_model.h5' ...
- DQN_Test_Node (for gazebo)
- driving_node (for real)

## Algorithms

### DQN (Deep Q-value Network)
CNN + Behavior Replay + Target Network
Enhance Behavior based on '_Previous behavior_' and '_Target theta_' in loop
<img width="780" height="518" alt="image" src="https://github.com/user-attachments/assets/76ba57f6-be9d-4501-b649-c13f365cb76a" />

## Logics - Agent
Select action, Make reward in **environment node** and apply to **Score** each step

### Finetuning
DQN_agent Improvements
1. Add _gc (garbage collector)_ to prevent memory leaking
2. Add _Model Checkpoints_ and Update automatically by **high-score**.

## Logs
Whole Improvements logs

* 1.10 Add Model Checkpoints on _agent.py_
* 1.11 Model Test #1
* 1.12 First Improving Reward logic on _envrionment.py_
* 1.13 Enhancing Reward logic on _envrionment.py_ & Model Test #2
* 1.14 Second Enhancing Reward logic on _envrionmnet.py_
* 1.15 Add _'YOLO detecting'_ on _environment.py_
* 1.16 Model Test #3
* 1.17 Third Enhancing Reward logic on _envrionmnet.py_
* 1.18 Model Test #4

## References
> https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning
> https://ai-com.tistory.com/entry/RL-%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-1-DQN-Deep-Q-Network
