PYTHON="/home/asus/miniconda3/envs/robocasa/bin/python"
HOST="127.0.0.1"
PORT="8865"
N_EPISODES=20
N_ENVS=5
RESULTS_DIR="./test_results_$(date +%Y%m%d_%H%M%S)"

TASKS=(
    "PnPCupToDrawerClose"
    "PnPPotatoToMicrowaveClose"
    "PnPMilkToMicrowaveClose"
    "PnPBottleToCabinetClose"
    "PnPWineToCabinetClose"
    "PnPCanToDrawerClose"
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA"
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA"
    "PosttrainPnPNovelFromCuttingboardToPanSplitA"
    "PosttrainPnPNovelFromCuttingboardToPotSplitA"
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA"
    "PosttrainPnPNovelFromPlacematToBasketSplitA"
    "PosttrainPnPNovelFromPlacematToBowlSplitA"
    "PosttrainPnPNovelFromPlacematToPlateSplitA"
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA"
    "PosttrainPnPNovelFromPlateToBowlSplitA"
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA"
    "PosttrainPnPNovelFromPlateToPanSplitA"
    "PosttrainPnPNovelFromPlateToPlateSplitA"
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA"
    "PosttrainPnPNovelFromTrayToPlateSplitA"
    "PosttrainPnPNovelFromTrayToPotSplitA"
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA"
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA"
)

mkdir -p "$RESULTS_DIR"
echo "Task,Success_Rate" > "$RESULTS_DIR/results.csv"

# 测试所有任务
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    echo "[$((i+1))/${#TASKS[@]}] Testing: $task"
    
    $PYTHON scripts/simulation_service.py \
        --client --websocket \
        --host "$HOST" --port "$PORT" \
        --env_name "gr1_unified/${task}_GR1ArmsAndWaistFourierHands_Env" \
        --video_dir "./videos/$task" \
        --max_episode_steps 720 \
        --n_envs "$N_ENVS" \
        --n_episodes "$N_EPISODES" \
        2>&1 | tee "$RESULTS_DIR/${task}.log"
    
    # 提取成功率
    rate=$(grep "Success rate:" "$RESULTS_DIR/${task}.log" | tail -1 | awk '{print $3}')
    echo "$task,$rate" >> "$RESULTS_DIR/results.csv"
    echo "  -> Success: $rate"
done

# 生成报告
{
    echo "=== GR00T GR1 Evaluation Results ==="
    echo "Date: $(date)"
    echo "Episodes: $N_EPISODES | Envs: $N_ENVS"
    echo ""
    column -t -s',' "$RESULTS_DIR/results.csv"
    echo ""
    avg=$(awk -F',' 'NR>1 {sum+=substr($2,1,length($2)-1); n++} END {printf "%.2f%%", sum/n}' "$RESULTS_DIR/results.csv")
    echo "Average Success Rate: $avg"
} | tee "$RESULTS_DIR/summary.txt"

echo ""
echo "Results saved to: $RESULTS_DIR"