# 📄 README: Job `prune_deit_4.2G_10989270`

## 🔧 Job Summary

This job performed structured pruning on the model **`deit_base_distilled_patch16_224`**, reducing its size and computational cost while maintaining core architectural features.

The pruning was done using Taylor pruning criteria with specified head and dimension pruning ratios, then saved as a pruned checkpoint for later evaluation and fine-tuning.

---

## ⚙️ Configuration

- **Model**: `deit_base_distilled_patch16_224`
- **Pruning type**: Taylor pruning
- **Pruning ratio**: 0.5
- **Head pruning ratio**: 0.5
- **Head dimension pruning ratio**: 0.25
- **Global pruning**: Enabled
- **Batch sizes**: Train & validation batch size: 64
- **Taylor batch size for gradients**: 50
- **Output checkpoint**:  
  `/work/hdd/bewo/mahdi/Isomorphic-Pruning/output/pruned/deit_4.2G.pth`

---

## 💻 Hardware and SLURM settings

- **GPUs**: 1 A100
- **CPUs per task**: 32
- **Memory**: 32 GB
- **Time limit**: 1 hour
- **Partition**: gpuA100x4-interactive

---

## ✅ Results

### Model pruning summary

- **MACs**: Reduced from **17.68 G** to **4.15 G**
- **Parameters**: Reduced from **87.34 M** to **20.67 M**
- **Final checkpoint saved**: `/work/hdd/bewo/mahdi/Isomorphic-Pruning/output/pruned/deit_4.2G.pth`

---

## 📄 Logs

- [SLURM error log](logs/slurm/prune_deit_4.2G_10989270.err)
- [SLURM output log](logs/slurm/prune_deit_4.2G_10989270.out)

---

## 💬 Notes

- The job successfully finished pruning and saved the pruned model.
- Next steps include evaluating the pruned model using your `evaluate.py` script to measure accuracy and further fine-tuning if necessary.
