"""
数据探索脚本
分析数据集的基本信息、标签分布、文本长度、图像尺寸等
"""
import os
import pandas as pd
from PIL import Image
from collections import Counter

def explore_dataset():
    """探索数据集基本信息"""
    # 读取训练数据
    train_df = pd.read_csv(r'D:\当代人工智能\project5\train.txt', sep=',', header=None, names=['guid', 'tag'])
    test_df = pd.read_csv(r'D:\当代人工智能\project5\test_without_label.txt', sep=',', header=None, names=['guid', 'tag'])
    
    print("=" * 60)
    print("数据集统计信息")
    print("=" * 60)
    print(f"训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}")
    print(f"\n标签列类型分布:")
    print(train_df['tag'].value_counts())
    
    # 检查数据文件
    data_dir = r'D:\当代人工智能\project5\data'
    all_files = os.listdir(data_dir)
    txt_files = [f for f in all_files if f.endswith('.txt')]
    jpg_files = [f for f in all_files if f.endswith('.jpg')]
    
    print(f"\n文本文件数量: {len(txt_files)}")
    print(f"图像文件数量: {len(jpg_files)}")
    
    # 检查配对情况
    print(f"\n检查数据配对情况...")
    missing_text = 0
    missing_image = 0
    for guid in train_df['guid'].head(50):
        txt_path = os.path.join(data_dir, f"{guid}.txt")
        img_path = os.path.join(data_dir, f"{guid}.jpg")
        if not os.path.exists(txt_path):
            missing_text += 1
        if not os.path.exists(img_path):
            missing_image += 1
    print(f"前50个样本中缺失文本: {missing_text}, 缺失图像: {missing_image}")
    
    # 检查图像尺寸分布
    print(f"\n分析图像尺寸分布（采样100张）...")
    img_sizes = []
    sample_guids = train_df['guid'].head(100).tolist()
    for guid in sample_guids:
        img_path = os.path.join(data_dir, f"{guid}.jpg")
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                img_sizes.append(img.size)
            except Exception as e:
                print(f"警告: 无法读取图像 {guid}.jpg - {e}")
    
    if img_sizes:
        size_counter = Counter(img_sizes)
        print(f"最常见的图像尺寸 (Top 5):")
        for size, count in size_counter.most_common(5):
            print(f"  {size}: {count}张")
        
        widths = [s[0] for s in img_sizes]
        heights = [s[1] for s in img_sizes]
        print(f"宽度范围: {min(widths)} ~ {max(widths)}, 平均: {sum(widths)/len(widths):.1f}")
        print(f"高度范围: {min(heights)} ~ {max(heights)}, 平均: {sum(heights)/len(heights):.1f}")
    
    # 检查文本长度
    print(f"\n分析文本长度分布（采样100条）...")
    text_lengths = []
    text_samples = []
    for guid in sample_guids:
        txt_path = os.path.join(data_dir, f"{guid}.txt")
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    text_lengths.append(len(text))
                    if len(text_samples) < 3:
                        text_samples.append(text[:100])
            except Exception as e:
                print(f"警告: 无法读取文本 {guid}.txt - {e}")
    
    if text_lengths:
        print(f"文本长度统计:")
        print(f"  最短: {min(text_lengths)} 字符")
        print(f"  最长: {max(text_lengths)} 字符")
        print(f"  平均: {sum(text_lengths)/len(text_lengths):.1f} 字符")
        print(f"  中位数: {sorted(text_lengths)[len(text_lengths)//2]} 字符")
        
        print(f"\n文本样例（前3条）:")
        for i, sample in enumerate(text_samples, 1):
            print(f"  样例{i}: {sample}...")
    
    print("=" * 60)
    print("数据探索完成")
    print("=" * 60)
    
    return train_df, test_df

if __name__ == "__main__":
    explore_dataset()
