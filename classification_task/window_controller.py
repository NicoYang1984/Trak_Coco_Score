import os
# Disable flash attention
os.environ["TORCH_NVCC_FLAGS"] = "-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import gc
import sys
import psutil
import shutil
import numpy as np
import torch as ch
# Disable flash attention backends immediately after importing torch
ch.backends.cuda.enable_flash_sdp(False)
ch.backends.cuda.enable_mem_efficient_sdp(False)
ch.backends.cuda.enable_math_sdp(True)  # Force math implementation
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, Subset
from pycocotools.coco import COCO
from PIL import Image
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import logging
from threading import Lock

# Must set environment variables before importing TRAK
os.environ["PYTORCH_DISABLE_PER_OP_AUTOCAST"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from trak import TRAKer

# Configuration parameters
COCO_ROOT = "/root/autodl-tmp/datasets/coco"
TRAIN_ANNOTATION = f"{COCO_ROOT}/annotations/instances_train2017.json"
VAL_ANNOTATION = f"{COCO_ROOT}/annotations/instances_val2017.json"
TRAIN_IMG_DIR = f"{COCO_ROOT}/train/images/train2017"
VAL_IMG_DIR = f"{COCO_ROOT}/val/images/val2017"
TRAIN_SET_SIZE = 118287
VAL_SET_SIZE = 5000
WINDOW_SIZE = 300
TOP_K = 10
SAVE_ROOT = "/root/autodl-tmp/results/coco_classification_scores"
BATCH_SIZE = 4

# Thread-safe output management
print_lock = Lock()
def thread_safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def setup_device(gpu_id):
    """设置设备，使用指定的GPU ID"""
    if gpu_id is not None:
        # 设置当前进程使用的GPU
        device = f"cuda:{gpu_id}"
        # 设置CUDA设备
        ch.cuda.set_device(device)
    else:
        # 如果没有指定GPU ID，使用默认GPU
        device = "cuda" if ch.cuda.is_available() else "cpu"
    
    # 验证设备设置
    if ch.cuda.is_available():
        actual_device = f"cuda:{ch.cuda.current_device()}"
        device_name = ch.cuda.get_device_name()
        thread_safe_print(f"🎯 使用设备: {actual_device} ({device_name})")
    else:
        thread_safe_print(f"🎯 使用设备: {device}")
    
    return device

# Custom COCO dataset class
class COCODataset:
    def __init__(self, root, annotation, img_dir, transform=None):
        thread_safe_print(f"🔄 Loading COCO dataset: {annotation}")
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root
        self.img_dir = img_dir
        self.transform = transform
        
        self.cat_id_to_name = {}
        categories = self.coco.loadCats(self.coco.getCatIds())
        for cat in categories:
            self.cat_id_to_name[cat['id']] = cat['name']
        
        thread_safe_print(f"✅ Dataset loaded: {len(self.ids)} samples, {len(self.cat_id_to_name)} categories")

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        img = Image.open(img_path).convert('RGB')
        
        cat_ids = list(set(ann['category_id'] for ann in anns))
        cat_names = []
        
        for cat_id in cat_ids:
            if cat_id in self.cat_id_to_name:
                cat_names.append(self.cat_id_to_name[cat_id])
            else:
                cat_names.append("object")
        
        if not cat_names:
            text = "a photo"
        else:
            text = f"a photo of {', '.join(cat_names)}"
        
        if not cat_ids:
            text = "a photo"
            label = 0
        else:
            text = f"a photo of {', '.join(cat_names)}"

            # Use COCO's built-in continuous category mapping
            raw_category_id = cat_ids[0]

            # Get all COCO category IDs and map to continuous indices
            all_cat_ids = self.coco.getCatIds()
            try:
                label = all_cat_ids.index(raw_category_id)
            except ValueError:
                label = 0
                thread_safe_print(f"⚠️ Unexpected COCO category ID: {raw_category_id}, using default label 0")
        return img, text, label

    def __len__(self):
        return len(self.ids)

def init_resnet18_coco(device):
    """Initialize ResNet-18 for COCO classification (80 classes)"""
    import torchvision.models as models
    import torchvision.transforms as transforms
    
    # Load pre-trained ResNet-18 model
    model = models.resnet18(weights="DEFAULT")
    
    # Adapt for COCO's 80-class classification task
    num_ftrs = model.fc.in_features
    model.fc = ch.nn.Linear(num_ftrs, 80)  # COCO has 80 classes
    
    # COCO dataset preprocessing (same as ImageNet)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    # Set to evaluation mode and move to target device
    model = model.to(device).eval()
    
    thread_safe_print("✅ ResNet-18 model loaded for COCO classification")
    return model, preprocess

def get_gpu_memory(device):
    """Get GPU memory usage for specific device"""
    if 'cuda' in device:
        try:
            allocated = ch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = ch.cuda.memory_reserved(device) / (1024 ** 2)
            return allocated, reserved
        except Exception as e:
            thread_safe_print(f"⚠️ 无法获取GPU内存信息: {e}")
            return 0, 0
    return 0, 0

def clean_memory(objs: list, device=None):
    """Clean memory/VRAM"""
    if objs is not None:
        if not isinstance(objs, (list, tuple)):
            objs = [objs]
        for obj in objs:
            if isinstance(obj, TRAKer):
                for attr in ['_gradient_computer', '_projector', '_features']:
                    if hasattr(obj, attr):
                        delattr(obj, attr)
            del obj
    gc.collect()
    if device and 'cuda' in device:
        try:
            ch.cuda.synchronize(device)
            ch.cuda.empty_cache()
            ch.cuda.ipc_collect()
            allocated, reserved = get_gpu_memory(device)
            thread_safe_print(f"[VRAM Status] allocated: {allocated:.2f}MB, reserved: {reserved:.2f}MB")
        except Exception as e:
            thread_safe_print(f"⚠️ GPU清理时出错: {e}")

def run_classification_task(window_idx, train_loader, val_loader, window_indices, device, task_id="CLASS"):
    """Run classification task"""
    thread_safe_print(f"🎯 [{task_id}] Starting classification task for window {window_idx} on {device}")
    start_time = time.time()
    
    try:
        # Create save directory for classification task
        classification_save_dir = f"{SAVE_ROOT}/window_{window_idx}"
        os.makedirs(classification_save_dir, exist_ok=True)
        result_path = f"{classification_save_dir}/top_contrib_samples.csv"
        
        if os.path.exists(result_path):
            thread_safe_print(f"📁 [{task_id}] Classification task for window {window_idx} already completed")
            return True
        
        # Initialize classification model in its own CUDA context
        with ch.cuda.device(device if 'cuda' in device else None):
            thread_safe_print(f"🔄 [{task_id}] Loading ResNet-18 model...")
            model, preprocess = init_resnet18_coco(device)
            thread_safe_print(f"✅ [{task_id}] Model loaded, device: {device}, precision: {next(model.parameters()).dtype}")

            # Wrap dataset to return labels instead of text
            class ClassificationWrapper:
                def __init__(self, original_dataset):
                    self.original_dataset = original_dataset
                    self.num_classes = 80  # COCO has 80 classes
                    
                def __getitem__(self, index):
                    image, text, label = self.original_dataset[index]

                    if label < 0 or label >= 80:
                        thread_safe_print(f"⚠️ Invalid label {label} at index {index}, clamping to valid range")
                        label = max(0, min(79, label))

                    return preprocess(image), label
                    
                def __len__(self):
                    return len(self.original_dataset)

            # Create classification datasets
            train_classification_dataset = ClassificationWrapper(Subset(train_loader.dataset, range(len(train_loader.dataset))))
            val_classification_dataset = ClassificationWrapper(Subset(val_loader.dataset, range(len(val_loader.dataset))))
            
            train_classification_loader = DataLoader(
                train_classification_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            val_classification_loader = DataLoader(
                val_classification_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )

            # Initialize TRAKer for classification
            use_half = next(model.parameters()).dtype == ch.float16
            traker = TRAKer(
                model=model,
                task="image_classification",
                train_set_size=len(window_indices),
                save_dir=classification_save_dir,
                device=device,
                proj_dim=1024,
                use_half_precision=use_half
            )

            # Featurize training samples
            thread_safe_print(f"🔄 [{task_id}] Starting training sample featurization...")
            checkpoint = model.state_dict()
            traker.load_checkpoint(checkpoint, model_id=0)
            
            # Featurization computation
            for batch_idx, (images, labels) in enumerate(tqdm(train_classification_loader, desc=f"[{task_id}] Featurizing", position=0, leave=True)):
                images = images.to(device)
                labels = labels.to(device)
                
                traker.featurize(batch=(images, labels), num_samples=images.shape[0])
                model.zero_grad(set_to_none=True)
            
            traker.finalize_features()
            thread_safe_print(f"✅ [{task_id}] Training sample featurization completed")

            # Clean training resources
            clean_memory([train_classification_loader], device)

            # Calculate contribution scores
            thread_safe_print(f"🔄 [{task_id}] Starting contribution score calculation...")
            traker.start_scoring_checkpoint(
                exp_name="coco_window",
                checkpoint=checkpoint,
                model_id=0,
                num_targets=VAL_SET_SIZE
            )
            
            # Calculate scores
            for batch_idx, (images, labels) in enumerate(tqdm(val_classification_loader, desc=f"[{task_id}] Scoring", position=1, leave=True)):
                images = images.to(device)
                labels = labels.to(device)
                
                traker.score(batch=(images, labels), num_samples=images.shape[0])
                model.zero_grad(set_to_none=True)
            
            scores = traker.finalize_scores(exp_name="coco_window")
            thread_safe_print(f"✅ [{task_id}] Contribution score calculation completed, score matrix shape: {scores.shape}")

            # Save top K high contribution samples
            if hasattr(scores, 'cpu'):
                scores_np = scores.cpu().numpy()
            else:
                scores_np = scores
            
            avg_scores = scores_np.mean(axis=1)
            top_indices = np.argsort(avg_scores)[-TOP_K:][::-1]
            
            with open(result_path, "w") as f:
                f.write("original_train_idx,avg_contribution_score\n")
                for idx in top_indices:
                    original_idx = window_indices[idx]
                    f.write(f"{original_idx},{avg_scores[idx]:.12f}\n")
            
            thread_safe_print(f"✅ [{task_id}] Top {TOP_K} samples saved to {result_path}")

            # Final cleanup
            clean_memory([traker, scores, model, val_classification_loader], device)
        
        total_time = time.time() - start_time
        thread_safe_print(f"✅ [{task_id}] Classification task for window {window_idx} completed, total time: {total_time:.2f} seconds")
        return True
        
    except Exception as e:
        thread_safe_print(f"❌ [{task_id}] Classification task for window {window_idx} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_single_window(window_idx, gpu_id):
    """Process single window with classification task only"""
    device = setup_device(gpu_id)
    thread_safe_print(f"🔄 Process {os.getpid()} starting processing for window {window_idx} on {device}")
    
    # Window save directory
    window_save_dir = f"{SAVE_ROOT}/window_{window_idx}"
    os.makedirs(window_save_dir, exist_ok=True)
    
    # Check if task is already completed
    classification_result_path = f"{window_save_dir}/top_contrib_samples.csv"
    
    if os.path.exists(classification_result_path):
        thread_safe_print(f"📁 Window {window_idx} classification task already completed")
        return True
    
    start_time = time.time()
    
    try:
        # Load datasets
        thread_safe_print("🔄 Loading datasets...")
        
        # Calculate current window range
        start_idx = window_idx * WINDOW_SIZE
        end_idx = min((window_idx + 1) * WINDOW_SIZE, TRAIN_SET_SIZE)
        window_indices = list(range(start_idx, end_idx))
        actual_window_size = len(window_indices)
        thread_safe_print(f"Window range: training samples [{start_idx}, {end_idx - 1}], actual size: {actual_window_size}")

        # Load training set window subset
        train_dataset_full = COCODataset(
            root=COCO_ROOT,
            annotation=TRAIN_ANNOTATION,
            img_dir=TRAIN_IMG_DIR,
        )
        train_dataset_subset = Subset(train_dataset_full, window_indices)
        train_loader = DataLoader(
            train_dataset_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Load validation set
        val_dataset = COCODataset(
            root=COCO_ROOT,
            annotation=VAL_ANNOTATION,
            img_dir=VAL_IMG_DIR,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Run classification task
        thread_safe_print("🚀 Starting classification task execution...")
        classification_success = run_classification_task(window_idx, train_loader, val_loader, window_indices, device, "CLASSIFICATION")
        
        # Final cleanup
        clean_memory([train_loader, val_loader, train_dataset_subset, train_dataset_full, val_dataset], device)
        
        total_time = time.time() - start_time
        
        if classification_success:
            thread_safe_print(f"✅ Window {window_idx} processing completed, total time: {total_time:.2f} seconds")
            return True
        else:
            thread_safe_print(f"❌ Window {window_idx} processing failed")
            return False
        
    except Exception as e:
        thread_safe_print(f"❌ Window {window_idx} processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python classification_window_worker.py <window_index> <gpu_id>")
        sys.exit(1)
    
    window_idx = int(sys.argv[1])
    gpu_id = int(sys.argv[2])
    
    success = process_single_window(window_idx, gpu_id)
    sys.exit(0 if success else 1)
