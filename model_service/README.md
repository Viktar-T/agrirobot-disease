# model_service/ — piece 4: model service v0

H8 §4 piece 4, detailed in H8 §6 · H6 E15 · owner M · W1–W6

One service behind one interface — `(frame + metadata) → (scores, uncertainty, abstain/unknown, model version)` — backed by cached frozen DINOv2 (champion) / DINOv3 (challenger) features, small heads and an abstention rule trained on the open bean sets; plus the evaluation harness and a thin ROS 2 client.

**Done when**: the acceptance criteria of H8 §6.11 are met.
