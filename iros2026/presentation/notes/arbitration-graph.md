We use arbitration graphs to separate responsibilities and increase transparency.
So-called behavior components propose actions,
a shared verifier rejects unsafe proposals,
a cost arbitrator picks the best proposal at runtime,
and a fallback component is executed if no proposal is safe.

Here, we combine the learning-based `FlowDrive` with the rule-based `PDM-Closed` planner.
We then evaluate this architecture using nuPlan and interplan,
commonly used benchmarks for closed-loop planning in autonomous driving.
