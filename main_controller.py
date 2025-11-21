import subprocess
import sys
import time
import os
import argparse
from multiprocessing import Pool, cpu_count

# 配置COCO数据集相关路径和参数
SAVE_ROOT = "/root/autodl-tmp/results/coco_classification_scores"
TRAIN_SET_SIZE = 118287  # COCO 2017训练集样本数
WINDOW_SIZE = 300        # 每个窗口包含的训练样本数（根据内存调整）
TOTAL_WINDOWS = (TRAIN_SET_SIZE + WINDOW_SIZE - 1) // WINDOW_SIZE  # 总窗口数

# GPU配置
NUM_GPUS = 8  # L40s服务器有8张GPU
MAX_CONCURRENT_PROCESSES = NUM_GPUS  # 同时运行的最大进程数，每个进程使用一个GPU

def check_window_completed(window_idx):
    """检查指定窗口是否已完成（通过结果文件判断）"""
    result_path = f"{SAVE_ROOT}/window_{window_idx}/top_contrib_samples.csv"
    return os.path.exists(result_path)

def run_single_window(args):
    """在独立进程中运行单个窗口处理任务"""
    window_idx, gpu_id = args
    print(f"🚀 启动进程处理窗口 {window_idx} (GPU {gpu_id})")
    start_time = time.time()
    
    try:
        # 设置环境变量指定GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # 调用窗口处理脚本，传递窗口索引
        process = subprocess.Popen([
            sys.executable,
            "/workspace/trak/coco_score_classification/window_controller.py",
            str(window_idx),
            str(gpu_id)  # 传递GPU ID给子进程
        ], env=env)
        
        returncode = process.wait()
        elapsed = time.time() - start_time
        
        if returncode == 0:
            print(f"✅ 窗口 {window_idx} 处理成功 (GPU {gpu_id}, 用时: {elapsed:.2f}秒)")
            return (window_idx, True, gpu_id)
        else:
            print(f"❌ 窗口 {window_idx} 处理失败 (GPU {gpu_id}, 退出码: {returncode})")
            return (window_idx, False, gpu_id)
            
    except Exception as e:
        print(f"💥 窗口 {window_idx} 进程异常 (GPU {gpu_id}): {e}")
        return (window_idx, False, gpu_id)

def run_parallel_windows(windows_to_process, max_workers=None):
    """并行处理多个窗口"""
    if max_workers is None:
        max_workers = min(MAX_CONCURRENT_PROCESSES, len(windows_to_process))
    
    print(f"🔄 启动并行处理，使用 {max_workers} 个进程")
    
    # 为每个窗口分配GPU（轮询分配）
    window_gpu_pairs = []
    for i, window_idx in enumerate(windows_to_process):
        gpu_id = i % NUM_GPUS
        window_gpu_pairs.append((window_idx, gpu_id))
    
    results = []
    with Pool(processes=max_workers) as pool:
        try:
            # 使用imap_unordered以获得更好的进度反馈
            for result in pool.imap_unordered(run_single_window, window_gpu_pairs):
                results.append(result)
                completed = len([r for r in results if r[1]])
                total = len(windows_to_process)
                print(f"📊 进度: {completed}/{total} 完成")
        except KeyboardInterrupt:
            print("⚠️ 用户中断，等待当前进程完成...")
            pool.terminate()
            pool.join()
            raise
    
    return results

def main():
    parser = argparse.ArgumentParser(description='多GPU COCO TRAK处理控制器')
    parser.add_argument("--start", type=int, default=0, help="起始窗口索引")
    parser.add_argument("--end", type=int, default=TOTAL_WINDOWS, help="结束窗口索引")
    parser.add_argument("--gpus", type=int, default=NUM_GPUS, help="使用的GPU数量")
    parser.add_argument("--parallel", type=int, default=MAX_CONCURRENT_PROCESSES, 
                       help="并行进程数")
    parser.add_argument("--sequential", action="store_true", 
                       help="顺序执行而非并行执行")
    args = parser.parse_args()
    
    global NUM_GPUS, MAX_CONCURRENT_PROCESSES
    NUM_GPUS = args.gpus
    MAX_CONCURRENT_PROCESSES = min(args.parallel, NUM_GPUS)
    
    total_windows = args.end - args.start
    completed_count = 0
    failed_windows = []
    pending_windows = []
    
    print(f"🎯 开始处理窗口范围: [{args.start}, {args.end-1}] (共{total_windows}个窗口)")
    print(f"🖥️  使用 {NUM_GPUS} 个GPU，最大并行度: {MAX_CONCURRENT_PROCESSES}")
    os.makedirs(SAVE_ROOT, exist_ok=True)
    
    # 检查已完成窗口并收集待处理窗口
    for window_idx in range(args.start, args.end):
        if check_window_completed(window_idx):
            print(f"📁 窗口 {window_idx} 已完成，跳过")
            completed_count += 1
        else:
            pending_windows.append(window_idx)
    
    print(f"🔄 待处理窗口: {len(pending_windows)} 个")
    
    if not pending_windows:
        print("✅ 所有窗口已完成!")
        return
    
    # 选择执行模式
    if args.sequential:
        print("🔀 使用顺序执行模式")
        # 顺序执行
        for i, window_idx in enumerate(pending_windows):
            print(f"\n{'='*50}")
            print(f"🔄 开始处理窗口 {i+1}/{len(pending_windows)} (总进度: {completed_count+i+1}/{total_windows})")
            print(f"{'='*50}")
            
            gpu_id = i % NUM_GPUS
            success = run_single_window((window_idx, gpu_id))[1]
            if success:
                completed_count += 1
            else:
                failed_windows.append(window_idx)
            
            time.sleep(1)  # 进程间延迟
    else:
        print("⚡ 使用并行执行模式")
        # 并行执行
        results = run_parallel_windows(pending_windows)
        
        for window_idx, success, gpu_id in results:
            if success:
                completed_count += 1
            else:
                failed_windows.append(window_idx)
    
    # 输出总结
    print(f"\n{'='*50}")
    print("🎊 处理完成!")
    print(f"✅ 成功: {completed_count}/{total_windows} 个窗口")
    if failed_windows:
        print(f"❌ 失败: {len(failed_windows)} 个窗口: {failed_windows}")
        # 保存失败窗口列表以便重试
        with open(f"{SAVE_ROOT}/failed_windows.txt", "w") as f:
            for window in failed_windows:
                f.write(f"{window}\n")
        print(f"💾 失败窗口列表已保存到: {SAVE_ROOT}/failed_windows.txt")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
