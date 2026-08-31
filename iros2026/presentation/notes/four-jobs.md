It's one custom pipeline handling multiple responsibilities.

It **generates** trajectories
and **selects** among proposals.
It **verifies** the output to make sure it is safe
and it falls back to e.g. an emergency stop if it isn't.

All of this happens inside this pipeline,
which means the components are tightly coupled, hard to reason about and cannot be reused.

Hi, I'm Nick and together with my colleagues I propose a more organized approach to this problem.
