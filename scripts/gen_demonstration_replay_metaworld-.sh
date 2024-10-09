# bash scripts/gen_demonstration_metaworld.sh basketball
# task_names=("basketball" "coffee-pull" "coffee-push" "bin-picking" "box-close" "soccer" "disassemble" "reach")
# task_names=("disassemble" "reach")
task_names=("dial-turn" "door-lock" "door-open" "door-unlock" "handle-pull-side" "lever-pull" "pegunplugside" "basketball" "bin-picking" "coffee-pull" "coffee-push" "hammer" "soccer" "sweep" "sweep-into" "hand-insert" "pick-out-of-hole" "pick-place" "push" "push-back" "shelf-place" "stick-pull" "stick-push" "pick-place-wall") 

# 进入指定目录
cd third_party/Metaworld

# 指定可见的 CUDA 设备
export CUDA_VISIBLE_DEVICES=1

# 循环遍历每个任务
for task_name in "${task_names[@]}"
do
    # 打印当前执行的任务名称
    echo "Running task: ${task_name}"

    # 尝试不同的 speed 参数
    for speed in 2 3 4
    do
        echo "Running task: ${task_name} with speed: ${speed}"

        # 执行 Python 脚本，每个任务运行一次
        python gen_demonstration_expert_replay.py --env_name=${task_name} \
            --num_episodes 20 \
            --root_dir "../../3D-Diffusion-Policy/data/" \
            --method 4 \
            --speed ${speed}
        
        # 检查上一个命令是否执行成功，如果失败则退出循环
        if [ $? -ne 0 ]; then
            echo "Error occurred in task: ${task_name} with speed: ${speed}. Exiting..."
            exit 1
        fi
    done
done

echo "All tasks completed!"
