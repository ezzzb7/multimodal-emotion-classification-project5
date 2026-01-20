# =========================================================================
# 多模态情感分类 - 一键运行所有实验脚本
# =========================================================================
# 功能：
# 1. 继续基线训练（从10轮到早停，最多50轮）
# 2. 运行Text-Only消融实验（50轮）
# 3. 运行Image-Only消融实验（50轮）
# 4. 运行Early Fusion高级融合（50轮）
# 5. 运行Cross-Attention高级融合（50轮）
# =========================================================================

param(
    [switch]$SkipBaseline,      # 跳过基线继续训练
    [switch]$QuickTest,         # 快速测试模式（每个10轮）
    [switch]$OnlyAblation,      # 只运行消融实验
    [switch]$OnlyFusion         # 只运行融合策略
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# 颜色输出函数
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-ColorOutput ("="*80) "Cyan"
    Write-ColorOutput $Text "Cyan"
    Write-ColorOutput ("="*80) "Cyan"
    Write-Host ""
}

function Write-Success {
    param([string]$Text)
    Write-ColorOutput "✓ $Text" "Green"
}

function Write-Error {
    param([string]$Text)
    Write-ColorOutput "✗ $Text" "Red"
}

function Write-Info {
    param([string]$Text)
    Write-ColorOutput "→ $Text" "Yellow"
}

# 记录开始时间
$scriptStartTime = Get-Date
$resultsFile = "experiment_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

# 实验结果列表
$experimentResults = @()

# 配置文件路径
$configPath = "configs\config.py"
$backupConfigPath = "configs\config.py.backup"

# 备份原始配置
Write-Info "备份原始配置文件..."
Copy-Item $configPath $backupConfigPath -Force

# 修改配置的函数
function Update-Config {
    param(
        [string]$ModelType,
        [string]$FusionType = "late",
        [string]$Modality = "multimodal",
        [string]$ResumeFrom = "None",
        [int]$NumEpochs = 50
    )
    
    $content = Get-Content $configPath -Raw
    
    # 更新各项配置
    $content = $content -replace "MODEL_TYPE = '[^']*'", "MODEL_TYPE = '$ModelType'"
    $content = $content -replace "MODALITY = '[^']*'", "MODALITY = '$Modality'"
    $content = $content -replace "FUSION_TYPE = '[^']*'", "FUSION_TYPE = '$FusionType'"
    $content = $content -replace "NUM_EPOCHS = \d+", "NUM_EPOCHS = $NumEpochs"
    
    if ($ResumeFrom -eq "None") {
        $content = $content -replace "RESUME_FROM = .*", "RESUME_FROM = None"
    } else {
        $content = $content -replace "RESUME_FROM = .*", "RESUME_FROM = r'$ResumeFrom'"
    }
    
    Set-Content $configPath $content -NoNewline
}

# 运行实验的函数
function Run-Experiment {
    param(
        [string]$Name,
        [string]$ModelType,
        [string]$FusionType = "late",
        [string]$Modality = "multimodal",
        [string]$ResumeFrom = "None",
        [int]$NumEpochs = 50
    )
    
    Write-Header "实验: $Name"
    Write-Info "时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Info "配置: ModelType=$ModelType, Fusion=$FusionType, Modality=$Modality"
    Write-Info "轮数: $NumEpochs"
    if ($ResumeFrom -ne "None") {
        Write-Info "断点续传: $ResumeFrom"
    }
    Write-Host ""
    
    $expStartTime = Get-Date
    
    # 更新配置
    Update-Config -ModelType $ModelType -FusionType $FusionType -Modality $Modality `
                  -ResumeFrom $ResumeFrom -NumEpochs $NumEpochs
    
    # 运行训练
    try {
        python train.py
        $exitCode = $LASTEXITCODE
        
        $expDuration = (Get-Date) - $expStartTime
        $durationMin = [math]::Round($expDuration.TotalMinutes, 1)
        
        if ($exitCode -eq 0) {
            Write-Success "实验 '$Name' 完成！用时: $durationMin 分钟"
            $script:experimentResults += [PSCustomObject]@{
                Name = $Name
                Status = "成功"
                Duration = "$durationMin 分钟"
                Time = Get-Date -Format "HH:mm:ss"
            }
        } else {
            Write-Error "实验 '$Name' 失败！退出码: $exitCode"
            $script:experimentResults += [PSCustomObject]@{
                Name = $Name
                Status = "失败"
                Duration = "$durationMin 分钟"
                Time = Get-Date -Format "HH:mm:ss"
            }
        }
    }
    catch {
        Write-Error "实验 '$Name' 出错: $_"
        $script:experimentResults += [PSCustomObject]@{
            Name = $Name
            Status = "错误"
            Duration = "-"
            Time = Get-Date -Format "HH:mm:ss"
        }
    }
    
    Write-Host ""
    Write-ColorOutput ("-"*80) "Gray"
    Write-Host ""
}

# 主程序
Write-Header "多模态情感分类 - 完整实验流程"
Write-Info "开始时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Info "工作目录: $(Get-Location)"

if ($QuickTest) {
    Write-ColorOutput "【快速测试模式】每个实验训练10轮" "Magenta"
    $epochs = 10
} else {
    Write-ColorOutput "【完整训练模式】每个实验训练50轮" "Magenta"
    $epochs = 50
}

Write-Host ""
Write-Info "实验计划:"
if (-not $SkipBaseline -and -not $OnlyAblation -and -not $OnlyFusion) {
    Write-Host "  1. 基线Late Fusion（继续训练：epoch 11-$epochs）"
}
if (-not $OnlyFusion) {
    Write-Host "  2. Text-Only消融实验（$epochs 轮）"
    Write-Host "  3. Image-Only消融实验（$epochs 轮）"
}
if (-not $OnlyAblation) {
    Write-Host "  4. Early Fusion高级融合（$epochs 轮）"
    Write-Host "  5. Cross-Attention高级融合（$epochs 轮）"
}
Write-Host ""

# 询问确认
$confirmation = Read-Host "确认开始实验？(Y/N)"
if ($confirmation -ne 'Y' -and $confirmation -ne 'y') {
    Write-ColorOutput "已取消" "Yellow"
    # 恢复配置
    Copy-Item $backupConfigPath $configPath -Force
    Remove-Item $backupConfigPath
    exit
}

Write-Host ""
Write-ColorOutput "开始执行实验..." "Green"
Write-Host ""

# =========================================================================
# 实验 1: 基线Late Fusion（继续训练）
# =========================================================================
if (-not $SkipBaseline -and -not $OnlyAblation -and -not $OnlyFusion) {
    # 查找最新的基线checkpoint
    $baselineCheckpoint = Get-ChildItem "checkpoints" -Filter "late_multimodal_*_epoch10.pth" | 
                          Sort-Object LastWriteTime -Descending | 
                          Select-Object -First 1
    
    if ($baselineCheckpoint) {
        $checkpointPath = $baselineCheckpoint.FullName.Replace($PWD.Path + "\", "")
        Run-Experiment -Name "基线Late Fusion（继续训练）" `
                       -ModelType "multimodal" `
                       -FusionType "late" `
                       -Modality "multimodal" `
                       -ResumeFrom $checkpointPath `
                       -NumEpochs $epochs
    } else {
        Write-ColorOutput "警告: 未找到基线checkpoint，跳过继续训练" "Yellow"
    }
}

# =========================================================================
# 实验 2: Text-Only消融实验
# =========================================================================
if (-not $OnlyFusion) {
    Run-Experiment -Name "Text-Only消融实验" `
                   -ModelType "text_only" `
                   -FusionType "late" `
                   -Modality "text" `
                   -ResumeFrom "None" `
                   -NumEpochs $epochs
}

# =========================================================================
# 实验 3: Image-Only消融实验
# =========================================================================
if (-not $OnlyFusion) {
    Run-Experiment -Name "Image-Only消融实验" `
                   -ModelType "image_only" `
                   -FusionType "late" `
                   -Modality "image" `
                   -ResumeFrom "None" `
                   -NumEpochs $epochs
}

# =========================================================================
# 实验 4: Early Fusion高级融合
# =========================================================================
if (-not $OnlyAblation) {
    Run-Experiment -Name "Early Fusion高级融合" `
                   -ModelType "multimodal" `
                   -FusionType "early" `
                   -Modality "multimodal" `
                   -ResumeFrom "None" `
                   -NumEpochs $epochs
}

# =========================================================================
# 实验 5: Cross-Attention高级融合
# =========================================================================
if (-not $OnlyAblation) {
    Run-Experiment -Name "Cross-Attention高级融合" `
                   -ModelType "multimodal" `
                   -FusionType "cross_attention" `
                   -Modality "multimodal" `
                   -ResumeFrom "None" `
                   -NumEpochs $epochs
}

# =========================================================================
# 总结
# =========================================================================
$totalDuration = (Get-Date) - $scriptStartTime
$totalHours = [math]::Round($totalDuration.TotalHours, 2)
$totalMinutes = [math]::Round($totalDuration.TotalMinutes, 1)

Write-Header "实验总结"

# 显示结果表格
Write-Host ""
$experimentResults | Format-Table -AutoSize
Write-Host ""

# 统计
$successCount = ($experimentResults | Where-Object { $_.Status -eq "成功" }).Count
$totalCount = $experimentResults.Count

Write-ColorOutput "总实验数: $totalCount" "Cyan"
Write-ColorOutput "成功: $successCount" "Green"
Write-ColorOutput "失败: $($totalCount - $successCount)" "Red"
Write-ColorOutput "总用时: $totalHours 小时 ($totalMinutes 分钟)" "Cyan"

# 保存结果到文件
$experimentResults | Format-Table -AutoSize | Out-File $resultsFile
Write-Host ""
Write-Info "结果已保存到: $resultsFile"

# 列出所有生成的checkpoint
Write-Host ""
Write-Header "生成的模型Checkpoint"
Get-ChildItem "checkpoints" -Filter "best_*.pth" | ForEach-Object {
    Write-Host "  - $($_.Name)" -ForegroundColor Green
}

# 下一步提示
Write-Host ""
Write-Header "下一步操作"
Write-Host ""
Write-Info "1. 生成可视化图表:"
Write-Host "   Get-ChildItem logs | ForEach-Object { python utils\visualize.py `"logs\`$(`$_.Name)`" }"
Write-Host ""
Write-Info "2. 选择最佳模型预测测试集:"
Write-Host "   python predict.py --checkpoint checkpoints\best_<最佳模型>.pth --output predictions.txt"
Write-Host ""
Write-Info "3. 对比所有实验结果:"
Write-Host "   python evaluate.py --compare-all"
Write-Host ""
Write-Info "4. 填写实验报告:"
Write-Host "   打开 EXPERIMENT_REPORT_TEMPLATE.md"
Write-Host ""

# 恢复配置文件
Write-Info "恢复原始配置文件..."
Copy-Item $backupConfigPath $configPath -Force
Remove-Item $backupConfigPath

Write-Header "所有实验完成！"
Write-ColorOutput "实验结束时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "Green"
Write-Host ""

# 询问是否立即生成可视化
$generateViz = Read-Host "是否立即为所有实验生成可视化图表？(Y/N)"
if ($generateViz -eq 'Y' -or $generateViz -eq 'y') {
    Write-Header "生成可视化图表"
    Get-ChildItem logs -Directory | ForEach-Object {
        Write-Info "生成图表: $($_.Name)"
        python utils\visualize.py "logs\$($_.Name)"
    }
    Write-Success "所有可视化图表生成完成！"
}

Write-Host ""
Write-ColorOutput "🎉 实验全部完成！祝论文/报告写作顺利！" "Green"
Write-Host ""
