for hidden_features in 32; do
for learning_rate in 1e-3; do
for depth in 3; do
for residual in 1; do
for weight_decay in 1e-8; do
for semantic_weight in -1.0; do
for num_heads in 4; do
for fc_dropout in 0.0 0.1 0.2 0.4; do
for a_h_dropout in 0.0 0.1 0.2 0.4; do
for a_x_dropout in 0.0 0.1 0.2 0.4; do

hidden_features=$hidden_features \
learning_rate=$learning_rate \
depth=$depth \
residual=$residual \
weight_decay=$weight_decay \
semantic_weight=$semantic_weight \
num_heads=$num_heads \
fc_dropout=$fc_dropout \
a_h_dropout=$a_h_dropout \
a_x_dropout=$a_x_dropout \
bsub < run.sh

done; done; done; done; done; done; done; done; done; done
