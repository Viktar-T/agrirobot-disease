# interface/ — piece 1: interface v0 (the minimal observation contract)

H8 §4 piece 1 · H6 E12 · owner P + E · W1–W2 (v0.1 in W10)

What the robot, or any stand-in for it, hands over per frame, and what it accepts back (trigger / waypoint request stubs). Delivered as a JSON Schema, a validator library used by every consumer, and ROS 2 message definitions.

**Done when** (H8): a synthetic frame validates, a frame missing any mandatory field is rejected with a named reason, and the robotics side has read and acknowledged the document.
