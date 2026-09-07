# Edge-source control training logs (Canny maps as stage-1 source)

canny_s1.log holds the complete stage-1 trajectory (epoch 1 loss 0.0051 -> 0.0000 by
epoch 50). canny_s0.log and canny_s2.log begin mid-stage-1 (epochs 34 and 33) because the
launcher reopens the log in write mode on a resume after a session disconnect; training
itself resumed from the per-epoch checkpoint and is unaffected -- only the text record of
the earlier epochs is lost. All three logs show stage 1 at loss 0.0000 for every recorded
epoch from the low thirties onward, and no errors or NaNs.

Reading: the binary Canny maps are reproduced exactly by the sigmoid output head through the
skip connections, so the stage-1 pretext task collapses (loss 0.0000 vs about 0.003 for the
MNIST source with the same code), and the source encoder is never forced to learn stroke
structure. That is the mechanism behind the poor transfer reported in the paper's source
control.
