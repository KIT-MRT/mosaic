Here I brought one qualitative example to illustrate how this works.

We see the ego controlled by PDM-Open approaching an accident.
PDM is a lange-following planner
so it handles the situation as expected and gets stuck behind the accident.

---

This is an out-of-distribution scenario that FlowDrive has never seen during training.
It turns unstable triggering the internal emergency stop repeatedly, also getting stuck behind the accident.

---

Mosaic composes exactly these two planners, allowing it to stabilize the approach using PDM,
giving FlowDrive a chance to propose a lane change to get around the accident.

---

If that example left you with a question or two,
come see me at my poster.
