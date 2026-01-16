ALL_DATASET_PATHS=(
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "/mnt/project/public/world_model/RawData/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000"
)

python scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 8 --batch-size 60 --learning_rate 3e-5 \
  --output-dir /mnt/project/world_model/baseline_checkpoints/GR00T_N1_5_finetuned_robocasa \
  --data-config fourier_gr1_arms_waist --embodiment_tag gr1 \
  --tune-visual \
  --max-steps 30000 --save-steps 5000