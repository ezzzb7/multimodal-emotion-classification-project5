"""
Simple test script for Stage 1: Data Processing Module
Memory-optimized version for Windows
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_stage1():
    """Test data loading with minimal memory footprint"""
    
    print("\n" + "="*70)
    print("STAGE 1: DATA PROCESSING TEST (Memory Optimized)")
    print("="*70)
    
    # Paths
    data_dir = r'D:\当代人工智能\project5\data'
    train_label = r'D:\当代人工智能\project5\train.txt'
    test_label = r'D:\当代人工智能\project5\test_without_label.txt'
    
    # Step 1: Test imports
    print("\n[1] Testing imports...")
    try:
        from data.preprocessing import TextPreprocessor, ImagePreprocessor
        from data.dataset import MultimodalDataset
        from data.data_loader import create_data_splits
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Step 2: Create splits
    print("\n[2] Creating train/val splits...")
    try:
        if not os.path.exists('splits/train_split.txt'):
            create_data_splits(train_label, val_ratio=0.15, seed=42)
        print("✓ Splits ready")
    except Exception as e:
        print(f"✗ Split creation failed: {e}")
        return False
    
    # Step 3: Test single sample loading
    print("\n[3] Testing single sample loading...")
    try:
        text_prep = TextPreprocessor(remove_emoji=False, lowercase=True)
        img_prep = ImagePreprocessor(mode='train', img_size=224)
        
        dataset = MultimodalDataset(
            data_dir=data_dir,
            label_file='splits/train_split.txt',
            text_preprocessor=text_prep,
            image_preprocessor=img_prep,
            mode='train'
        )
        
        # Load first sample
        sample = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  - guid: {sample['guid']}")
        print(f"  - text: {sample['text'][:50]}...")
        print(f"  - image shape: {sample['image'].shape}")
        print(f"  - label: {sample['label']}")
        
    except Exception as e:
        print(f"✗ Sample loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test small batch loading
    print("\n[4] Testing small batch loading...")
    try:
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        batch = next(iter(loader))
        print(f"✓ Batch loaded successfully")
        print(f"  - batch size: {len(batch['guid'])}")
        print(f"  - image batch shape: {batch['image'].shape}")
        print(f"  - labels: {batch['label'].tolist()}")
        
    except Exception as e:
        print(f"✗ Batch loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test full data loaders (small batch)
    print("\n[5] Testing full data loaders (batch_size=4)...")
    try:
        from data.data_loader import get_data_loaders
        
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir=data_dir,
            train_label_file=train_label,
            test_label_file=test_label,
            batch_size=4,  # Small batch for memory
            val_ratio=0.15,
            num_workers=0,
            seed=42
        )
        
        print(f"✓ Data loaders created")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        # Test iteration
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        print(f"✓ Can iterate through loaders")
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("STAGE 1 TEST PASSED ✓")
    print("="*70 + "\n")
    return True


if __name__ == "__main__":
    try:
        success = test_stage1()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
