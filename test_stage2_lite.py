"""
Lightweight test for Stage 2 - Memory optimized
Tests model components with production settings (feature_dim=512)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import gc


def clear_cache():
    """Clear memory between tests"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_imports():
    """Test 1: Just imports"""
    print("\n[Test 1] Imports...")
    try:
        from models.text_encoder import TextEncoder
        from models.image_encoder import ImageEncoder
        from models.fusion import LateFusion
        from models.multimodal_model import MultimodalClassifier
        print("✓ Imports OK")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_text_encoder_only():
    """Test 2: Text encoder alone"""
    print("\n[Test 2] Text Encoder (downloading model if needed)...")
    try:
        from models.text_encoder import TextEncoder
        
        # Production settings
        encoder = TextEncoder(
            model_name='distilbert-base-uncased',
            output_dim=512,
            freeze_bert=True
        )
        
        texts = ["test"]
        with torch.no_grad():
            features = encoder(texts)
        
        print(f"✓ Text encoder OK: {features.shape}")
        print(f"  Feature dim: 512 (production setting)")
        
        # Clean up
        del encoder
        clear_cache()
        return True
        
    except Exception as e:
        print(f"✗ Text encoder failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_encoder_only():
    """Test 3: Image encoder alone"""
    print("\n[Test 3] Image Encoder...")
    try:
        from models.image_encoder import ImageEncoder
        
        encoder = ImageEncoder(
            model_name='resnet50',
            output_dim=512,
            freeze_backbone=True,
            pretrained=True
        )
        
        images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = encoder(images)
        
        print(f"✓ Image encoder OK: {features.shape}")
        print(f"  Feature dim: 512 (production setting)")
        
        del encoder
        clear_cache()
        return True
        
    except Exception as e:
        print(f"✗ Image encoder failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion():
    """Test 4: Fusion"""
    print("\n[Test 4] Fusion modules...")
    try:
        from models.fusion import LateFusion
        
        fusion = LateFusion(512, 512)
        text_feat = torch.randn(1, 512)
        img_feat = torch.randn(1, 512)
        
        out = fusion(text_feat, img_feat)
        print(f"✓ Fusion OK: {out.shape}")
        print(f"  Output: 512+512=1024 (late fusion)")
        return True
        
    except Exception as e:
        print(f"✗ Fusion failed: {e}")
        return False


def test_full_model():
    """Test 5: Full model"""
    print("\n[Test 5] Full multimodal model (production config)...")
    print("  NOTE: This may take 1-2 minutes and use significant memory")
    try:
        from models.multimodal_model import MultimodalClassifier
        
        model = MultimodalClassifier(
            num_classes=3,
            fusion_type='late',
            feature_dim=512,
            freeze_encoders=True
        )
        
        texts = ["test"]
        images = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            logits = model(texts, images)
        
        print(f"✓ Full model OK: {logits.shape}")
        print(f"  Logits: {logits[0]}")
        print(f"  Architecture: DistilBERT(512) + ResNet50(512) -> Late Fusion(1024) -> Classifier(3)")
        
        # Count parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {total:,}")
        print(f"  Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
        
        del model
        clear_cache()
        return True
        
    except Exception as e:
        print(f"✗ Full model failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*70)
    print("STAGE 2: LIGHTWEIGHT TEST (Step-by-step)")
    print("="*70)
    
    # Check command line args for skipping full model test
    skip_full = '--skip-full' in sys.argv
    
    tests = [
        ("Imports", test_imports),
        ("Text Encoder", test_text_encoder_only),
        ("Image Encoder", test_image_encoder_only),
        ("Fusion", test_fusion),
    ]
    
    if not skip_full:
        tests.append(("Full Model", test_full_model))
    else:
        print("\nSkipping full model test (use without --skip-full to include)")
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            if not result:
                print(f"\n✗ {name} failed, stopping tests")
                break
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
            break
    
    print("\n" + "="*70)
    print("RESULTS:")
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:20s}: {status}")
    
    if skip_full:
        print("\n  Note: Full model test skipped (memory constrained)")
        print("  All components tested individually - Stage 2 complete!")
    
    print("="*70)
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✅ STAGE 2 PASSED - Baseline model components working!")
        print("\nNext: Stage 3 - Training pipeline")
    
    sys.exit(0 if all_passed else 1)
