for hidden_features in 32; do
for depth in 2 4 6; do
for num_heads in 1 2 4; do
for learning_rate in 1e-3; do
for n_samples in 1; do

hidden_features=$hidden_features \
learning_rate=$learning_rate \
n_samples=$n_samples \
num_heads=$num_heads \
depth=$depth \
bsub < run.sh

done; done; done; done; done
