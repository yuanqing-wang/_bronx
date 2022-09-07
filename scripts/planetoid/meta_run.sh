for hidden_features in 32 64 128; do
for learning_rate in 1e-3; do
for depth in 2 3 4; do
for weight_decay in 1e-5 1e-6 1e-7 1e-8; do
for semantic_weight in -1; do
for num_heads in 2 4; do

hidden_features=$hidden_features \
learning_rate=$learning_rate \
depth=$depth \
weight_decay=$weight_decay \
num_heads=$num_heads \
bsub < run.sh

done; done; done; done; done; done
