# bash scripts/gen_demonstration_replay_metaworld-.sh basketball
# task_names=( "soccer" "sweep" "sweep-into" "hand-insert" "pick-out-of-hole" "pick-place" "push" "push-back" "shelf-place" "stick-pull" "stick-push" "pick-place-wall")
# task_names=()
task_names=("basketball" "dial-turn" "door-lock" "door-open" "door-unlock" "handle-pull-side" "lever-pull" "peg-unplug-side" "bin-picking" "coffee-pull" "coffee-push" "soccer" "sweep" "sweep-into" "hand-insert" "pick-out-of-hole"  "pick-place" "push" "push-back" "shelf-place"  "stick-pull" "stick-push" "pick-place-wall") 
# if task that use gripper then-> method3 -> to make avg speed near 2x
# task_names=("Basketball" "bin-picking" "coffee-push" "coffee-pull"
# if task that no use gripper then —> method4
# training: use data that both fail or success, for u cannot know if a task is success after speed up?

# training: use data that only success, the previous training is using the success data.

# 进入指定目录
cd third_party/Metaworld

# 指定可见的 CUDA 设备
export CUDA_VISIBLE_DEVICES=1

# 循环遍历每个任务
for task_name in "${task_names[@]}"
do
    # 打印当前执行的任务名称
    echo "Running task: ${task_name}"

    # 尝试不同的 nmax 参数(只有当method = 4的时候这个nmax才有意义)
    for nmax in 5 
    do
        # 尝试不同的 speed 参数
        for speed in 2
        do
            echo "Running task: ${task_name} with speed: ${speed} and nmax: ${nmax}"

            # 执行 Python 脚本，每个任务运行一次
            python gen_demonstration_expert_replay-demo.py --env_name=${task_name} \
                --num_episodes 20 \
                --root_dir "../../3D-Diffusion-Policy/data/" \
                --method 4 \
                --speed ${speed} \
                --nmax ${nmax}
            
            # 检查上一个命令是否执行成功，如果失败则退出循环
            if [ $? -ne 0 ]; then
                echo "Error occurred in task: ${task_name} with speed: ${speed} and nmax: ${nmax}. Exiting..."
                exit 1
            fi
        done
    done
done

echo "All tasks completed!"
