"""
分析和比较不同配置的训练结果
"""

import os
import re
from pathlib import Path
import json

def extract_best_accuracy(log_file):
    """从日志文件提取最佳验证准确率"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找 "Best model at epoch X with val_acc: Y"
    matches = re.findall(r'Best model at epoch (\d+) with val_acc: ([\d.]+)', content)
    if matches:
        # 返回最后一个（最终的最佳）
        epoch, acc = matches[-1]
        return {'epoch': int(epoch), 'accuracy': float(acc)}
    
    return None

def get_checkpoint_info(checkpoint_dir):
    """获取checkpoint目录信息"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = list(Path(checkpoint_dir).glob('best_*.pth'))
    if not checkpoints:
        return None
    
    # 获取最新的checkpoint
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    size_mb = latest.stat().st_size / (1024 * 1024)
    
    return {
        'name': latest.name,
        'size_mb': round(size_mb, 2),
        'time': latest.stat().st_mtime
    }

def main():
    print("\n" + "="*70)
    print("训练结果分析 - Hyperparameter Tuning")
    print("="*70 + "\n")
    
    experiments = {
        'Original Baseline': {
            'log_dir': 'logs',
            'checkpoint_dir': 'checkpoints',
            'description': '冻结编码器 (FREEZE_ENCODERS=True)'
        },
        'Improved Baseline': {
            'log_dir': 'logs_improved',
            'checkpoint_dir': 'checkpoints_improved',
            'description': '解冻编码器 + 分层学习率'
        },
        'Aggressive': {
            'log_dir': 'logs_aggressive',
            'checkpoint_dir': 'checkpoints_aggressive',
            'description': '完全解冻 + 大学习率'
        },
        'Conservative': {
            'log_dir': 'logs_conservative',
            'checkpoint_dir': 'checkpoints_conservative',
            'description': '部分解冻 + 强正则化'
        }
    }
    
    results = []
    
    for exp_name, exp_config in experiments.items():
        print(f"正在分析: {exp_name}...")
        
        # 查找最新的日志文件
        log_dir = exp_config['log_dir']
        checkpoint_dir = exp_config['checkpoint_dir']
        
        log_files = []
        if os.path.exists(log_dir):
            log_files = sorted(Path(log_dir).glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
        
        best_acc = None
        if log_files:
            best_acc = extract_best_accuracy(str(log_files[0]))
        
        checkpoint_info = get_checkpoint_info(checkpoint_dir)
        
        results.append({
            'name': exp_name,
            'description': exp_config['description'],
            'best_accuracy': best_acc,
            'checkpoint': checkpoint_info
        })
    
    # 打印结果表格
    print("\n" + "="*70)
    print("实验结果对比")
    print("="*70 + "\n")
    
    print(f"{'实验名称':<20} {'最佳Epoch':<12} {'验证准确率':<15} {'提升幅度':<12}")
    print("-" * 70)
    
    baseline_acc = None
    for result in results:
        name = result['name']
        desc = result['description']
        
        if result['best_accuracy']:
            epoch = result['best_accuracy']['epoch']
            acc = result['best_accuracy']['accuracy']
            
            if name == 'Original Baseline':
                baseline_acc = acc
                improvement = '-'
            elif baseline_acc:
                improvement = f"+{(acc - baseline_acc):.4f}"
            else:
                improvement = '?'
            
            print(f"{name:<20} {epoch:<12} {acc:<15.4f} {improvement:<12}")
        else:
            print(f"{name:<20} {'N/A':<12} {'未完成':<15} {'-':<12}")
    
    print("-" * 70)
    print()
    
    # 打印详细信息
    print("="*70)
    print("详细配置信息")
    print("="*70 + "\n")
    
    for result in results:
        print(f"【{result['name']}】")
        print(f"  策略: {result['description']}")
        
        if result['best_accuracy']:
            acc = result['best_accuracy']
            print(f"  最佳Epoch: {acc['epoch']}")
            print(f"  验证准确率: {acc['accuracy']:.4f}")
        
        if result['checkpoint']:
            ckpt = result['checkpoint']
            print(f"  Checkpoint: {ckpt['name']}")
            print(f"  文件大小: {ckpt['size_mb']} MB")
        
        print()
    
    # 推荐
    if all(r['best_accuracy'] for r in results):
        best_result = max(results, key=lambda r: r['best_accuracy']['accuracy'])
        print("="*70)
        print("🏆 推荐配置")
        print("="*70)
        print(f"\n最佳配置: {best_result['name']}")
        print(f"验证准确率: {best_result['best_accuracy']['accuracy']:.4f}")
        print(f"策略: {best_result['description']}")
        print("\n建议使用此配置运行完整的5个模型对比实验。\n")
    
    # 保存结果到JSON
    output_file = 'results/hyperparameter_tuning_results.json'
    os.makedirs('results', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {output_file}\n")

if __name__ == '__main__':
    main()
