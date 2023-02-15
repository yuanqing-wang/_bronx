for hidden_features in 32; do
for depth in 2; do
for num_heads in 1; do
for learning_rate in 1e-3; do
for weight_decay in 1e-8; do
for n_samples in 1; do
for scale in 1.0 10.0 100.0; do

hidden_features=$hidden_features \
learning_rate=$learning_rate \
n_samples=$n_samples \
num_heads=$num_heads \
weight_decay=$weight_decay \
depth=$depth \
scale=$scale \
bsub < run.sh

done; done; done; done; done; done; done
