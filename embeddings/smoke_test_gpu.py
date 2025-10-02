#!/usr/bin/env python3
"""
Smoke test for GPU setup and CLIP model loading.
Tests:
1. GPU availability
2. PyTorch/CUDA setup
3. Transformers library compatibility
4. CLIP model loading
5. Sample image embedding generation
"""

import sys
import time

def test_gpu_availability():
    """Test if GPU is available."""
    print("\n" + "="*60)
    print("TEST 1: GPU Availability")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_transformers_import():
    """Test transformers library import."""
    print("\n" + "="*60)
    print("TEST 2: Transformers Library")
    print("="*60)
    
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        from transformers import CLIPProcessor, CLIPModel
        print("✅ CLIP imports successful")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_model_loading():
    """Test CLIP model loading."""
    print("\n" + "="*60)
    print("TEST 3: CLIP Model Loading")
    print("="*60)
    
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        model_name = "openai/clip-vit-base-patch32"
        print(f"Loading model: {model_name}")
        
        start = time.time()
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        load_time = time.time() - start
        
        print(f"✅ Model loaded in {load_time:.2f}s")
        print(f"✅ Processor type: {type(processor).__name__}")
        print(f"✅ Model type: {type(model).__name__}")
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"✅ Model moved to: {device}")
        
        return True, model, processor, device
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_sample_embedding():
    """Test generating a sample embedding."""
    print("\n" + "="*60)
    print("TEST 4: Sample Image Embedding")
    print("="*60)
    
    try:
        import torch
        from PIL import Image
        import numpy as np
        from transformers import CLIPProcessor, CLIPModel
        
        # Load model
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # Create a dummy image (RGB, 224x224)
        print("Creating dummy image (224x224 RGB)...")
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        # Process and embed
        print("Processing image...")
        with torch.no_grad():
            inputs = processor(images=dummy_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            start = time.time()
            outputs = model.get_image_features(**inputs)
            embed_time = time.time() - start
            
            embedding = outputs.cpu().numpy()[0]
        
        print(f"✅ Embedding generated in {embed_time*1000:.2f}ms")
        print(f"✅ Embedding shape: {embedding.shape}")
        print(f"✅ Embedding dtype: {embedding.dtype}")
        print(f"✅ Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        print(f"✅ Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Test batch processing of multiple images."""
    print("\n" + "="*60)
    print("TEST 5: Batch Processing")
    print("="*60)
    
    try:
        import torch
        from PIL import Image
        from transformers import CLIPProcessor, CLIPModel
        
        # Load model
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # Create batch of dummy images
        batch_size = 8
        print(f"Creating batch of {batch_size} images...")
        images = [Image.new('RGB', (224, 224), color='red') for _ in range(batch_size)]
        
        # Process batch
        print(f"Processing batch of {batch_size} images...")
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            start = time.time()
            outputs = model.get_image_features(**inputs)
            batch_time = time.time() - start
            
            embeddings = outputs.cpu().numpy()
        
        print(f"✅ Batch processed in {batch_time*1000:.2f}ms")
        print(f"✅ Throughput: {batch_size/batch_time:.2f} images/sec")
        print(f"✅ Embeddings shape: {embeddings.shape}")
        print(f"✅ Per-image time: {batch_time*1000/batch_size:.2f}ms")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("🔥 GPU SMOKE TEST - CLIP Image Embeddings")
    print("="*60)
    
    results = {}
    
    # Test 1: GPU availability
    results['gpu'] = test_gpu_availability()
    
    # Test 2: Transformers import
    results['transformers'] = test_transformers_import()
    
    # Test 3: CLIP model loading
    results['model_loading'] = test_clip_model_loading()[0]
    
    # Test 4: Sample embedding
    results['sample_embedding'] = test_sample_embedding()
    
    # Test 5: Batch processing
    results['batch_processing'] = test_batch_processing()
    
    # Summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! GPU setup is ready.")
        print("="*60)
        return 0
    else:
        print("❌ SOME TESTS FAILED! Check errors above.")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

