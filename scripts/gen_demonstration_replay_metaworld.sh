# bash scripts/gen_demonstration_metaworld.sh basketball
# task_names=("basketball" "coffee-pull" "coffee-push" "bin-picking" "box-close" "soccer" "disassemble" "reach")
# task_names=("disassemble" "reach")
task_names=("coffee-pull" "coffee-push" "bin-picking" "box-close" "soccer" "disassemble" "reach" "button-press-topdown" "button-press-topdown-wall" "button-press" "button-press-wall" "coffee-button" "dial-turn" "disassemble" "door-close" "door-lock" "door-open" "door-unlock" "hand-insert" "drawer-close" "drawer-open" 
            "faucet-open" 
            "faucet-close" 
            "hammer" 
            "handle-press-side" 
            "handle-press" 
            "handle-pull-side" 
            "handle-pull" 
            "lever-pull" 
            "peg-insert-side" 
            "pick-place-wall" 
            "pick-out-of-hole" 
            "push-back" 
            "push" 
            "pick-place" 
            "plate-slide" 
            "plate-slide-side" 
            "plate-slide-back" 
            "plate-slide-back-side" 
            "peg-unplug-side" 
            "stick-push" 
            "stick-pull" 
            "push-wall" 
            "reach-wall" 
            "shelf-place" 
            "sweep-into" 
            "sweep")


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
    for speed in 2
    do
        echo "Running task: ${task_name} with speed: ${speed}"

        # 执行 Python 脚本，每个任务运行一次
        python gen_demonstration_expert_replay2x.py --env_name=${task_name} \
            --num_episodes 20 \
            --root_dir "../../3D-Diffusion-Policy/data/" \
            --method 1 \
            --speed ${speed}
        
        # 检查上一个命令是否执行成功，如果失败则退出循环
        if [ $? -ne 0 ]; then
            echo "Error occurred in task: ${task_name} with speed: ${speed}. Exiting..."
            exit 1
        fi
    done
done

echo "All tasks completed!"
