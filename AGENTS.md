# AGENTS.md - Python Coding Guidelines

## 1. コーディングスタイル
本プロジェクトでは **[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)** をベースとして採用します。

### 例外事項
* **Line Length (行長):** 読みやすさと現代的なディスプレイ環境を考慮し、最大 **120文字** まで許容します。

### 基本的な考え方
* 可読性が高く、Pythonic（Pythonらしい）な記述を心がける。
* 型ヒント（Type Annotations）を積極的に活用する。
    * Python 3.14を使用するため、`typing.List`や`typing.Tuple`ではなく、標準の`list`, `tuple`などを使用する。
* DocstringはGoogleスタイルで記述する。
* コメントおよびDocstringは **英語** で記述する。

---

## 2. ツール (Linter / Formatter)
コードの品質管理には、高速なRust製ツールである **[Ruff](https://docs.astral.sh/ruff/)** を使用します。

また、静的型チェックには **[Mypy](https://mypy-lang.org/)** を使用し、Pythonの型ヒントの整合性を検証します。

---

## 3. 開発フロー
コードを作成・変更した後は、コミット前に必ず以下のコマンドを実行してください。これにより、フォーマットの適用と、自動修正可能なエラーの解決が同時に行われます。
ただし、以下のコマンドは`docker/run_docker.sh`で起動したコンテナの中で実行をお願いします(dockerが使えない場合はローカルで実行してください)

```bash
# フォーマットの適用、静的解析、型チェック
ruff format && ruff check --fix && mypy . && pytest
```

## 4. チェックリスト
- [ ] 1行が120文字以内に収まっているか？
- [ ] Google Styleの命名規則（関数は`snake_case`、クラスは`PascalCase`など）に従っているか？
- [ ] コメントとDocstringは英語で記述されているか？
- [ ] testを作成したか？
- [ ] コミット前に`ruff format && ruff check --fix && mypy . && pytest`を実行したか？

## 5. アウトプット言語
特に指示がない場合、すべての回答は **日本語** で作成してください。

---

## 6. GitHub Copilot 利用時の指針
GitHub Copilot を利用して生成・編集する場合も、本ガイドラインは **必ず遵守** してください。

### 6.1 生成内容の確認
- 生成されたコード/ドキュメントは **必ずレビュー** し、要件・仕様・品質基準に合致することを確認する。
- 外部ライブラリの追加が含まれる場合は、`pyproject.toml` の依存関係に反映し、ライセンスと互換性を確認する。
- 既存の設計・命名・公開APIとの整合性を保つ。不要なリファクタリングや改行/整形の大量変更は避ける。

### 6.2 セキュリティと機密情報
- APIキー、パスワード、トークンなどの **機密情報は埋め込まない**。
- 例外/ログに機密情報が含まれないよう配慮する。

### 6.3 テストと品質
- 変更内容に対応するテストを追加/更新する。
- 変更後に `ruff format && ruff check --fix && mypy . && pytest` を実行し、問題がないことを確認する。

### 6.4 コメント/Docstring
- Copilot が生成したコメント/Docstringも **英語** で記述する。
- 冗長なコメントは避け、意図・前提・例外条件を明確にする。
