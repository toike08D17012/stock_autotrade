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

Ruffは、従来の `black`, `isort`, `flake8` などの機能を統合したツールとして機能します。

また、静的型チェックには **[Mypy](https://mypy-lang.org/)** を使用し、Pythonの型ヒントの整合性を検証します。

---

## 3. 開発フロー
コードを作成・変更した後は、コミット前に必ず以下のコマンドを実行してください。これにより、フォーマットの適用と、自動修正可能なエラーの解決が同時に行われます。
ただし、以下のコマンドは`docker/run_docker.sh`で起動したコンテナの中で実行をお願いします

```bash
# フォーマットの適用、静的解析、型チェック
ruff format && ruff check --fix && mypy .
```
実行内容の内訳
`ruff format`: コードのスタイル（インデント、空白、改行など）を自動調整します。
`ruff check --fix`: 未使用のインポートの削除や、推奨されない記述の自動修正を行います。
`mypy .`: プロジェクト全体の型チェックを行い、整合性を検証します。

## 4. チェックリスト
- [ ] 1行が120文字以内に収まっているか？
- [ ] Google Styleの命名規則（関数は`snake_case`、クラスは`PascalCase`など）に従っているか？
- [ ] コメントとDocstringは英語で記述されているか？
- [ ] コミット前に`ruff format && ruff check --fix && mypy .`を実行したか？

## 5. アウトプット言語
特に指示がない場合、**すべてのアーティファクト** は **日本語** で作成してください。
これには以下が含まれますが、これらに限定されません。

- **Task (`task.md`)**
- **Implementation Plan (`implementation_plan.md`)**
- **Walkthrough (`walkthrough.md`)**
- その他、ユーザーへの報告や計画に関するドキュメント
