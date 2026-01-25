# Python Workspace Template

[English Versions](README_en.md) | 日本語

株式の自動取引のためのアルゴリズム開発と、バックテストの実行によるアルゴリズムの検証を目的としたレポジトリです。
Dev Container、uv、Ruff、Mypyを用いたモダンな開発環境を提供します。

## 機能・特徴

- **パッケージ管理**: `uv` を使用した高速な依存関係解決
- **開発環境**: Dev Container (`.devcontainer`) による一貫した環境
- **静的解析・フォーマット**: `ruff` による高速なLint/Format
- **型チェック**: `mypy` による静的型チェック
- **機械学習対応**: PyTorch (CPU/CUDA) の動的なインストール設定済み

## 使い方

### 1. Dev Container の起動
VS Code で本リポジトリを開き、推奨される拡張機能「Dev Containers」を使用してコンテナを起動してください。
`postCreateCommand` により、自動的に `uv sync` が実行され、環境がセットアップされます。

### 2. 依存関係の追加
パッケージを追加する場合は `uv` を使用します。

```bash
uv add <package_name>
```

### 3. 品質管理コマンド
本プロジェクトでは `GEMINI.md` に基づき、以下のコマンドでの品質チェックを推奨しています。

```bash
# フォーマット、Lint自動修正、型チェックを一括実行
ruff format && ruff check --fix && mypy .
```

## ディレクトリ構成

- `.devcontainer/`: Dev Container 設定 (VS Code用)
- `environments/python/`: Python プロジェクト定義 (`pyproject.toml` はここに配置)
- `GEMINI.md`: コーディング規約 (Google Style, Ruff設定など)
