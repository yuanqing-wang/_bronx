for hidden_features in 32 64 128; do
for learning_rate in 1e-2; do
for weight_decay in 1e-4 5e-4; do

hidden_features=$hidden_features \
learning_rate=$learning_rate \
weight_decay=$weight_decay \
bsub < run.sh

done; done; done;
