for hidden_features in 32; do
for learning_rate in 1e-2; do
for depth in 2; do
for weight_decay in 1e-12; do
for num_heads in 2; do
for sigma in 0.01 0.1 0.2 0.4 0.8; do
for alpha in 0.05; do
for n_samples in 4; do

hidden_features=$hidden_features \
learning_rate=$learning_rate \
depth=$depth \
weight_decay=$weight_decay \
num_heads=$num_heads \
sigma=$sigma \
alpha=$alpha \
n_samples=$n_samples \
bsub < run.sh

done; done; done; done; done; done; done; done
