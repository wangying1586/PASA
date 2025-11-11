#!/bin/bash
# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=3

# 设置参数
dataset=${1:-"ICBHI"}              # 默认使用ICBHI数据集，支持: SPRSound, ICBHI, CirCor
task_type=${2:-"22"}       # 默认使用22任务
use_adaptive_aug=${3:-"True"}      # 默认启用二阶段SAC
phase2_trigger_patience=${4:-"20"} # 第二阶段触发耐心值
phase2_trigger_threshold=${5:-"0.01"} # 第二阶段触发阈值
gamma=${6:-"0.95"}                 # SAC折扣因子

# 时间戳和日志设置
timestamp=$(date +"%Y%m%d_%H%M%S")
batch_size=32
feature_type="log-mel"
data_dir="./datasets"
log_dir="./logs"
mkdir -p $log_dir

# 构建日志文件名
if [ "$use_adaptive_aug" = "True" ]; then
    log_file="${log_dir}/${dataset}_Task${task_type}_bs${batch_size}_TwoPhaseSAC_P${phase2_trigger_patience}_T${phase2_trigger_threshold}_gamma${gamma}_${timestamp}.log"
    echo "Starting Two-Phase SAC-based adaptive augmentation training"
    echo "Strategy: Batch-level → Sample-level adaptive augmentation"
else
    log_file="${log_dir}/${dataset}_Task${task_type}_bs${batch_size}_NoAug_${timestamp}.log"
    echo "Starting training without augmentation"
fi

# SAC相关参数
sac_lr_actor=3e-4
sac_lr_critic=3e-4
sac_buffer_size=10000
sac_tau=0.005
n_magnitude_levels=5

echo "Dataset: ${dataset}"
echo "Task Type: ${task_type}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Augmentation Strategy: Two-Phase SAC"
echo ""

# 数据集特定信息
if [ "$dataset" = "CirCor" ]; then
    echo "🫀 CirCor DigiScope Heart Murmur Detection Dataset"
elif [ "$dataset" = "SPRSound" ]; then
    echo "🫁 SPRSound Lung Sound Dataset"
    echo "  - Task: ${task_type}"
elif [ "$dataset" = "ICBHI" ]; then
    echo "🫁 ICBHI2017 Lung Sound Dataset"
    echo "  - Task: ${task_type}"
fi

# 执行训练
python train.py \
    --dataset ${dataset} \
    --task_type ${task_type} \
    --data_dir ${data_dir} \
    --feature_type ${feature_type} \
    --batch_size ${batch_size} \
    --warmup_epoch 25 \
    --warmup_lr 0.001 \
    --epoch 300 \
    --lr 0.0001 \
    --early_stop True \
    --num_workers 1 \
    --pin_memory True \
    --prefetch_factor 2 \
    --use_adaptive_aug ${use_adaptive_aug} \
    --n_magnitude_levels ${n_magnitude_levels} \
    --gamma ${gamma} \
    --sac_lr_actor ${sac_lr_actor} \
    --sac_lr_critic ${sac_lr_critic} \
    --sac_buffer_size ${sac_buffer_size} \
    --sac_tau ${sac_tau} \
    --phase2_trigger_patience ${phase2_trigger_patience} \
    --phase2_trigger_threshold ${phase2_trigger_threshold} > "$log_file" 2>&1 &

echo ""
echo "✅ Training started successfully!"
echo "📝 Log file: $log_file"
echo "🔢 Process ID: $!"
echo ""

# 提供使用示例
echo "📖 Usage examples:"
#echo "  ./SAC_run_TwoStage_v2.sh ICBHI multiclass True 20 0.02 0.95    # ICBHI多分类，默认二阶段参数"
echo "  ./SAC_run_TwoStage_v2.sh SPRSound 12 True 15 0.015 0.90        # SPRSound任务12，更敏感的切换条件"
#echo "  ./SAC_run_TwoStage_v2.sh ICBHI binary True 25 0.03 0.99        # ICBHI二分类，较宽松的切换条件"
echo "  ./SAC_run_TwoStage_v2.sh CirCor heart_murmur True 20 0.01 0.95 # CirCor心脏杂音检测"
#echo "  ./SAC_run_TwoStage_v2.sh CirCor heart_murmur False             # CirCor无增强基线"
echo ""

# 实时监控日志的函数
monitor_log() {
    if [ -f "$log_file" ]; then
        echo " Monitoring log file: $log_file"
        echo " Press Ctrl+C to stop monitoring"
        echo "========================================"
        tail -f "$log_file"
    else
        echo " Log file not found: $log_file"
    fi
}

# 询问用户是否要监控日志
echo ""
read -p " Do you want to monitor the training log? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    monitor_log
else
    echo " Training is running in background. Check the log file periodically:"
    echo "   tail -f $log_file"
fi