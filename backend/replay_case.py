"""单病例回放 CLI（LS02）。

三种模式统一走 Runner 业务编排：
- fixture：注入的固定结果，无网络、无 torch/LangChain 导入；
- local：真实 ResNet18/GTE/Chroma，LLM 与搜索仍为注入；
- live：完整在线链路。

stdin 关闭仍可运行；输入缺失、模式冲突、输出目录非空、越界路径均以
非零退出码失败并留失败回执（trace/consumed/result）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from replay_support import (
    REASON_MODE_CONFLICT,
    TrialError,
    TrialModeError,
    build_fixture_services,
    build_real_services,
    load_case,
    load_config,
    prepare_output_dir,
    resolve_under_root,
    validate_image,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay_case",
        description="Lingshu 单病例试用回放入口（fixture/local/live）",
    )
    parser.add_argument("--mode", required=True, choices=("fixture", "local", "live"))
    parser.add_argument("--case", required=True, help="病例 JSON（仓库内相对路径）")
    parser.add_argument("--image", required=True, help="舌苔图片（仓库内相对路径）")
    parser.add_argument("--config", required=True, help="试用配置 YAML（仓库内相对路径）")
    parser.add_argument("--cassette", required=True, help="外部 fixture cassette JSON")
    parser.add_argument("--output-dir", required=True, help="产物输出目录（仓库内相对路径，必须为空）")
    return parser


def run(argv: list) -> int:
    root = repo_root()
    args = build_parser().parse_args(argv)

    output_dir = resolve_under_root(root, args.output_dir, label="--output-dir")
    case_path = resolve_under_root(root, args.case, label="--case")
    image_path = resolve_under_root(root, args.image, label="--image")
    config_path = resolve_under_root(root, args.config, label="--config")
    cassette_path = resolve_under_root(root, args.cassette, label="--cassette")

    case = load_case(case_path)
    config = load_config(config_path)
    if args.mode != config.mode:
        raise TrialModeError(
            f"--mode {args.mode} 与配置 trial.mode={config.mode} 冲突", REASON_MODE_CONFLICT
        )
    if config.case_id != case.case_id:
        raise TrialError(
            f"配置 case_id={config.case_id} 与病例 case_id={case.case_id} 不一致",
            "config_invalid",
        )
    case_image = resolve_under_root(root, case.image, label="case.image")
    if case_image != image_path:
        raise TrialError(
            f"--image {args.image} 与病例声明的图片 {case.image} 不一致",
            "config_invalid",
        )
    if config.mode == "fixture":
        if cassette_path != resolve_under_root(
            root, config.search["cassette"], label="search.cassette"
        ):
            raise TrialError(
                f"--cassette {args.cassette} 与配置 search.cassette={config.search['cassette']} 不一致",
                "config_invalid",
            )

    validate_image(image_path)
    prepare_output_dir(output_dir)

    if config.mode == "fixture":
        services = build_fixture_services(config, root, cassette_path)
    else:
        services = build_real_services(config, root)

    from agentsRunner import Runner

    runner = Runner(config=config, services=services, root=root)
    outcome = runner.run_case(case, image_path, output_dir)
    return 0 if outcome["success"] else 1


def main(argv: list | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # 非 tty 环境可能不支持 reconfigure
        pass
    try:
        return run(argv)
    except TrialError as exc:
        print(f"[replay-case] 失败（{exc.reason_code}）: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[replay-case] 被中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
