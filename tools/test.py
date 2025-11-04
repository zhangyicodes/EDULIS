# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
import time
import gc
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import torch
import torch.nn as nn

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_memory_usage(model):
    """计算模型内存占用"""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return param_size, buffer_size


class FPSTestHook:
    """FPS测试钩子 - 在真实测试过程中测量性能"""

    def __init__(self, max_test_samples=100):
        self.max_test_samples = max_test_samples
        self.inference_times = []
        self.start_time = None
        self.total_samples = 0

    def before_test_iter(self, runner, batch_idx, data_batch):
        """测试每个batch前调用"""
        if batch_idx < self.max_test_samples:
            self.start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        """测试每个batch后调用"""
        if batch_idx < self.max_test_samples and self.start_time is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            inference_time = end_time - self.start_time
            self.inference_times.append(inference_time)
            self.total_samples += len(data_batch['inputs'])

            # 每10个batch打印一次进度
            if (batch_idx + 1) % 10 == 0:
                avg_time = sum(self.inference_times[-10:]) / min(10, len(self.inference_times))
                current_fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"    已测试 {batch_idx + 1} 个batch, 最近FPS: {current_fps:.2f}")

    def get_fps_stats(self):
        """获取FPS统计信息"""
        if not self.inference_times:
            return None

        # 计算统计信息
        avg_time = sum(self.inference_times) / len(self.inference_times)
        min_time = min(self.inference_times)
        max_time = max(self.inference_times)

        fps_avg = 1.0 / avg_time if avg_time > 0 else 0
        fps_min = 1.0 / max_time if max_time > 0 else 0  # 最慢的FPS
        fps_max = 1.0 / min_time if min_time > 0 else 0  # 最快的FPS

        return {
            'avg_fps': fps_avg,
            'min_fps': fps_min,
            'max_fps': fps_max,
            'avg_latency_ms': avg_time * 1000,
            'min_latency_ms': min_time * 1000,
            'max_latency_ms': max_time * 1000,
            'total_samples': self.total_samples,
            'total_batches': len(self.inference_times)
        }


def benchmark_real_fps(runner, max_samples=100):
    """基于真实数据集测试FPS性能"""
    print(f"⚡ 开始真实数据FPS测试")
    print(f"  �� 最大测试样本数: {max_samples}")
    print(f"  �� 数据集: {runner.cfg.test_dataloader.dataset.type}")
    print(f"  �� 注意: 这是端到端的真实性能测试")

    # 创建FPS测试钩子
    fps_hook = FPSTestHook(max_samples)

    # 保存原始的测试循环
    original_test_loop = runner.test_loop

    # 包装测试循环以添加FPS测量
    class FPSTestLoop:
        def __init__(self, original_loop, fps_hook):
            self.original_loop = original_loop
            self.fps_hook = fps_hook

        def run(self):
            print("    �� 开始真实数据推理...")

            # 获取数据加载器
            dataloader = self.original_loop.dataloader
            model = self.original_loop.runner.model

            model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated() / 1024 ** 2

            # 手动运行测试循环
            for batch_idx, data_batch in enumerate(dataloader):
                if batch_idx >= self.fps_hook.max_test_samples:
                    break

                self.fps_hook.before_test_iter(None, batch_idx, data_batch)

                # 执行推理
                with torch.no_grad():
                    outputs = model.test_step(data_batch)

                self.fps_hook.after_test_iter(None, batch_idx, data_batch, outputs)

            # GPU内存统计
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                memory_usage = peak_memory - initial_memory
                return memory_usage

            return 0

    # 运行FPS测试
    fps_test_loop = FPSTestLoop(original_test_loop, fps_hook)
    memory_usage = fps_test_loop.run()

    # 获取结果
    fps_stats = fps_hook.get_fps_stats()

    if fps_stats:
        print(f"\n�� 真实FPS测试结果:")
        print(f"  �� 平均FPS: {fps_stats['avg_fps']:.2f}")
        print(f"  ⚡ 最快FPS: {fps_stats['max_fps']:.2f}")
        print(f"  �� 最慢FPS: {fps_stats['min_fps']:.2f}")
        print(f"  ⏱️  平均延迟: {fps_stats['avg_latency_ms']:.1f}ms")
        print(f"  �� 延迟范围: {fps_stats['min_latency_ms']:.1f}ms - {fps_stats['max_latency_ms']:.1f}ms")
        print(f"  �� 测试样本: {fps_stats['total_samples']} 张图片")
        print(f"  �� 测试批次: {fps_stats['total_batches']} 个batch")
        if torch.cuda.is_available():
            print(f"  �� 推理显存: {memory_usage:.1f}MB")

        # 性能评级
        avg_fps = fps_stats['avg_fps']
        print(f"\n�� 性能评级:")
        if avg_fps >= 30:
            rating = "优秀 �� (实时应用)"
        elif avg_fps >= 15:
            rating = "良好 �� (交互应用)"
        elif avg_fps >= 5:
            rating = "一般 �� (批处理)"
        else:
            rating = "需优化 ⚠️"

        print(f"  {rating}")


    else:
        print("❌ FPS测试失败")
        return None


def format_number(num):
    """格式化大数字显示"""
    if num >= 1e12:
        return f"{num / 1e12:.2f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.2f}G"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(int(num))


def analyze_model(model, input_shape=(1, 3, 640, 640), runner=None, test_real_fps=True, max_fps_samples=100):
    """分析模型"""
    print("\n" + "=" * 70)
    print("�� MODEL ANALYSIS")
    print("=" * 70)
    print(f"��️ 模型类型: {type(model).__name__}")
    print(f"�� 输入尺寸: {input_shape}")
    print(f"�� 设备: {next(model.parameters()).device}")
    print("-" * 70)

    # 参数统计
    total_params, trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params

    print(f"�� 参数统计:")
    print(f"  总参数数量: {format_number(total_params)} ({total_params:,})")
    print(f"  可训练参数: {format_number(trainable_params)} ({trainable_params:,})")
    print(f"  冻结参数: {format_number(frozen_params)} ({frozen_params:,})")

    # 参数占比
    if total_params > 0:
        trainable_ratio = trainable_params / total_params * 100
        frozen_ratio = frozen_params / total_params * 100
        print(f"  可训练比例: {trainable_ratio:.1f}%")
        print(f"  冻结比例: {frozen_ratio:.1f}%")

    # 模型内存占用
    param_memory, buffer_memory = get_model_memory_usage(model)
    total_model_memory = (param_memory + buffer_memory) / (1024 ** 2)  # MB

    print(f"\n�� 内存占用:")
    print(f"  参数内存: {param_memory / (1024 ** 2):.2f} MB")
    print(f"  缓冲内存: {buffer_memory / (1024 ** 2):.2f} MB")
    print(f"  模型总内存: {total_model_memory:.2f} MB")

    # 模型复杂度分析
    print(f"\n�� 模型复杂度分析:")
    if total_params < 1e6:
        complexity = "轻量级"
        emoji = "��"
    elif total_params < 10e6:
        complexity = "小型"
        emoji = "��"
    elif total_params < 50e6:
        complexity = "中型"
        emoji = "��"
    elif total_params < 100e6:
        complexity = "大型"
        emoji = "��️"
    else:
        complexity = "超大型"
        emoji = "��️"

    print(f"  {emoji} 模型规模: {complexity}")

    # 与常见模型对比
    print(f"\n�� 与常见模型对比:")
    model_comparisons = [
        ("MobileNetV2", 3.5e6),
        ("ResNet-50", 25.6e6),
        ("YOLOv5s", 7.2e6),
        ("YOLOv8m", 25.9e6),
        ("Mask R-CNN", 44.2e6),
    ]

    for name, params in model_comparisons:
        if abs(total_params - params) / params < 0.2:  # 相差20%以内
            ratio = total_params / params
            print(f"  �� 接近 {name} (x{ratio:.2f})")
            break

    # 真实FPS性能测试
    if test_real_fps and runner is not None:
        print(f"\n" + "-" * 70)
        fps_stats = benchmark_real_fps(runner, max_fps_samples)
    else:
        print(f"\n�� 跳过FPS测试 (需要提供runner和数据集)")

    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test with real FPS analysis')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='work directory')
    parser.add_argument('--out', type=str, help='output file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='show directory')
    parser.add_argument('--wait-time', type=float, default=2)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--analyze-model', action='store_true', help='analyze model parameters and real FPS')
    parser.add_argument('--input-shape', nargs=4, type=int, default=[1, 3, 640, 640])
    parser.add_argument('--no-fps', action='store_true', help='skip FPS testing')
    parser.add_argument('--max-fps-samples', type=int, default=100, help='max samples for FPS test')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--only-analyze', action='store_true', help='只分析模型，不进行完整测试')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 设置work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        default_work_dir = osp.join('./work_dirs', 'enhanced_test')
        cfg.work_dir = default_work_dir

    os.makedirs(cfg.work_dir, exist_ok=True)
    print(f"�� 工作目录: {cfg.work_dir}")

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [dict(type='RandomFlip', prob=1.), dict(type='RandomFlip', prob=0.)],
                    [dict(type='PackDetInputs',
                          meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                     'scale_factor', 'flip', 'flip_direction'))],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    try:
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)
    except Exception as e:
        print(f"❌ 构建runner失败: {e}")
        # 备用方案
        if not hasattr(cfg, 'work_dir'):
            cfg.work_dir = './work_dirs/enhanced_test'
        if not hasattr(cfg, 'log_level'):
            cfg.log_level = 'INFO'
        if not hasattr(cfg, 'load_from'):
            cfg.load_from = args.checkpoint
        runner = Runner.from_cfg(cfg)

    # 模型分析
    if args.analyze_model:
        model = runner.model
        input_shape = tuple(args.input_shape)
        test_fps = not args.no_fps
        analyze_model(model, input_shape, runner if test_fps else None, test_fps, args.max_fps_samples)

    # 如果只分析模式，跳过完整测试
    if args.only_analyze:
        print("✅ 仅分析模式，跳过完整数据集测试")
        return

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), 'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=args.out))

    # start testing
    print("�� 开始完整数据集测试...")
    runner.test()
    print("✅ 测试完成")


if __name__ == '__main__':
    main()