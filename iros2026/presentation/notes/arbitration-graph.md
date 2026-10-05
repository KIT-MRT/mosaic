In Mosaic, we use arbitration graphs to separate responsibilities and increase transparency.
So-called behavior components propose trajectories.
These are your planners, anything that generates a trajectory;
rule-based, learned, hybrid or even end-to-end.
A shared verifier rejects unsafe proposals,
while a cost arbitrator picks the best option at runtime.
A fallback component is executed if no safe proposal is left.

Here, we combine the learning-based `FlowDrive` with the rule-based `PDM-Closed` planner
and evaluate this architecture using the nuPlan and interPlan benchmarks,
commonly used benchmarks for closed-loop planning in autonomous driving.
