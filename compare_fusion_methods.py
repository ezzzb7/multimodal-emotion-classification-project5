"""
对比不同融合方式的性能
避免数据泄漏：使用训练集bad case
"""

import subprocess
import sys


def run_experiment(fusion_type, use_augmented=False, **kwargs):
    """运行单个实验"""
    cmd = [
        sys.executable, 
        'train_improved_fusion_v2.py',
        '--fusion_type', fusion_type,
    ]
    
    if use_augmented:
        cmd.append('--use_augmented')
    
    # 添加其他参数
    for key, value in kwargs.items():
        cmd.extend([f'--{key}', str(value)])
    
    print(f"\n{'='*70}")
    print(f"实验: {fusion_type.upper()} Fusion ({'With' if use_augmented else 'Without'} Augmentation)")
    print('='*70)
    print(f"命令: {' '.join(cmd)}\n")
    
    subprocess.run(cmd, check=True)


def main():
    print("🔬 融合方法对比实验")
    print("="*70)
    print("实验配置:")
    print("  - 加强正则化: dropout=0.4, weight_decay=0.01")
    print("  - 梯度裁剪: 1.0")
    print("  - Early stopping: patience=5")
    print("  - 数据: 避免验证集泄漏，使用训练集bad case")
    print("="*70)
    
    experiments = [
        # 1. Attention Fusion (当前最佳)
        {
            'fusion_type': 'attention',
            'use_augmented': True,
            'dropout': 0.4,
            'weight_decay': 0.01,
            'patience': 5
        },
        
        # 2. Gated Fusion
        {
            'fusion_type': 'gated',
            'use_augmented': True,
            'dropout': 0.4,
            'weight_decay': 0.01,
            'patience': 5
        },
        
        # 3. Multi-Head Attention Fusion
        {
            'fusion_type': 'multihead',
            'use_augmented': True,
            'dropout': 0.4,
            'weight_decay': 0.01,
            'patience': 5
        },
    ]
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n\n{'#'*70}")
        print(f"# 实验 {i}/{len(experiments)}")
        print('#'*70)
        
        try:
            run_experiment(**exp_config)
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验失败: {e}")
            continue
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断实验")
            break
    
    print("\n\n" + "="*70)
    print("✓ 所有实验完成！")
    print("="*70)
    print("\n请查看各实验目录下的 history.csv 对比结果")


if __name__ == '__main__':
    main()
