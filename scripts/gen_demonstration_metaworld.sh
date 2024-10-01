#!/bin/bash
# bash scripts/gen_demonstration_metaworld.sh

# 定义任务名称的数组
# task_names=("basketball" "assembly" "coffee-pull" "coffee-push" "bin-picking" "box-close" "soccer" "disassemble" "reach")
task_names=("basketball")
# 进入指定目录
cd third_party/Metaworld

# 指定可见的 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 循环遍历每个任务
for task_name in "${task_names[@]}"
do
    # 打印当前执行的任务名称
    echo "Running task: ${task_name}"

    # 执行 Python 脚本，每个任务运行一次
    python gen_demonstration_expert_video.py --env_name="${task_name}" \
        --num_episodes 20 \
        --root_dir "../../3D-Diffusion-Policy/data/"
    
    # 检查上一个命令是否执行成功，如果失败则退出循环
    if [ $? -ne 0 ]; then
        echo "Error occurred in task: ${task_name}. Exiting..."
        exit 1
    fi
done

echo "All tasks completed!"
