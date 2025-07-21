# Constrained TRPO CarRacer

In my previous [TRPO CarRacer](https://github.com/nathanjjohnson7/TRPO_CarRacer) implementation, the episode was terminated immediately when the car left the track. Here, we do not end the episode but rather attempt to constrain the car to stay on the track by formulating the problem as a Constrained Markov Decsion Process (CMPD). Our first approach to solve the CMDP was the Constrained Policy Optimization (CPO) algorithm ([Achiam et al., 2017](https://arxiv.org/pdf/1705.10528)), which unfortunately did not perform as expected (details provided below). We then attempted to use Constraint-Rectified Policy Optimization (CRPO) ([Xu et al., 2020](https://arxiv.org/pdf/2011.058690)), which yielded signficantly better results.

We use the same base algorithm (TRPO) and modified environment as shown in the [TRPO CarRacer](https://github.com/nathanjjohnson7/TRPO_CarRacer) repository. 

### CPO Results
<img width="600" height="375" alt="cpo_training_log_graph" src="https://github.com/user-attachments/assets/a05c38ca-43c4-4167-96e0-382abe468b62" />

### CRPO Results
<img width="600" height="375" alt="crpo_training_log_graph" src="https://github.com/user-attachments/assets/01310ac3-e571-4fc6-b095-fc7367cbfe2a" />

https://github.com/user-attachments/assets/6fc062a6-3524-4d2d-b229-775d612da2c6

https://github.com/user-attachments/assets/b9217c92-7628-4fe2-bcf3-c81479521dfc

https://github.com/user-attachments/assets/3abef758-b9f1-4344-8aa0-1d93f89abae4

### Evaluation

CPO uses a hard constraint. Every time the constraint is even slightly violated (e.g. just 1 frame spent outside the track), a constraint minimization step is executed. This heavy-handed approach significantly hurts exploration since it is likely for the car to leave the track as it searches for high-reward actions. Each violation causes the algorithm to completely disregard performance as it attempts to reduce the constraint cost, drastically degrading performance. 

Furthermore, when the constraint is not being violated, the CPO algorithm tries to maximize rewards, subject to the track constraint. After multiple constraint minimization steps, reward maximizaion is significantly hampered since the model is wary of violating the constraint, preventing recovery.

Due to a combination of these factors, eventually our CPO algorithm begins to favor not moving at all, in order to avoid constraint costs.

Conversely, CRPO uses a soft constraint, which is more forgiving. A constraint minimization step is only executed if the average constraint cost advantage exceeds the specified threshold. Moreover, CRPO disregards the constraint when trying to maximize rewards, preventing exploration from becoming overly cautious, even after multiple constraint minimizations.

