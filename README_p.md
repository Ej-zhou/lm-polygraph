```
huggingface-cli login
```

Go under this directory
```
/mnt/nas_home/yz926/lm-polygraph
```

and run the HYDRA file
```
HYDRA_CONFIG=/mnt/nas_home/yz926/lm-polygraph/examples/configs/TEST_COQA.yaml  python ./scripts/polygraph_eval       save_path="./workdir/output"  
```

on HPC:
```
conda activate lmp
HYDRA_CONFIG=/home/yz926/lm-polygraph/examples/configs/TEST_COQA.yaml  python ./scripts/polygraph_eval       save_path="./workdir/output"  
```

Using slurm we should do

```
sbatch benchmark.slurm 
```



------
for LTL cluster:
```
watch -n 1 squeue
```


for HPC:
To check in on the queue
```
watch -n 10 squeue -u yz926
```

To get an interactive node
```
sintr -A KORHONEN-SL3-GPU -p ampere -N1 -n1 -t 0:20:0 --qos=INTR --gres=gpu:1
```