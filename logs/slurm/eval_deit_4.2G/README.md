# 📄 README: Job `eval_deit_4.2G_10989330`

## 🔧 Job Summary

This job evaluated the previously pruned model **`deit_base_distilled_patch16_224`** (checkpoint `deit_4.2G.pth`) on the ImageNet validation set. The evaluation assessed the model’s accuracy and loss after pruning.

---

## ⚙️ Configuration

- **Model**: `deit_base_distilled_patch16_224.fb_in1k`
- **Checkpoint**: `/work/hdd/bewo/mahdi/Isomorphic-Pruning/output/pruned/deit_4.2G.pth`
- **Data path**: `/work/hdd/bewo/ssoma1/imagenet_data`
- **Validation resize**: 256
- **Interpolation**: bicubic
- **Batch size**: Validation batch size: 64

---

## ✅ Results

- **MACs**: 4.15 G
- **Parameters**: 20.67 M
- **Top-1 Accuracy**: 0.27% (`0.0027`)
- **Validation Loss**: 6.7481

---

## 📄 Logs

- [Error log](logs/slurm/eval_deit_4.2G_10989330.err)
- [Output log](logs/slurm/eval_deit_4.2G_10989330.out)

---

## 💬 Notes

- The very low accuracy indicates that the pruned model likely requires fine-tuning or further calibration on the dataset to recover performance after pruning.
- Next steps may include fine-tuning the pruned checkpoint on ImageNet and re-evaluating.