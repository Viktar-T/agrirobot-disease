# emulator/ — piece 2: robot emulator

H8 §4 piece 2 · H6 E13 (and E11) · owner E · W2–W4 (software), W7–W8 (physical bags)

A ROS 2 node that replays MCAP bags and synthesises missions from a mission file on the robot's future topics, using the piece-1 messages. In W7–W8 a lab camera on a hand mount with a marker pose tag records real bags that go through the same path.

**Done when** (H8): one virtual mission and one real bag replay through the validator into the store and the frames are indistinguishable in schema.
