# Claude Code 指示書：main.tex を elsarticle でコンパイル

## やること

`papers/fugitive/main.tex` を pdflatex でコンパイルして PDF を生成する。
エラーが出たら修正してコンパイルが通るまで繰り返す。

## セットアップ

1. main.tex をリポジトリの `papers/fugitive/` に置く（ユーザーが claude.ai からコピー）
2. figure2_phase_diagram.pdf も同じディレクトリに置く
3. elsarticle.cls が必要。なければ `tlmgr install elsarticle` か CTAN からダウンロード

## コンパイル手順

```bash
cd papers/fugitive
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex  # 2回目（相互参照解決）
```

## エラーが出た場合

main.tex は pandoc で markdown → LaTeX 変換した本文部分を含んでいる。
以下の種類のエラーが出る可能性がある：

1. **Too many }'s** — frontmatter コマンドの括弧ずれ。elsarticle の `\begin{frontmatter}` ... `\end{frontmatter}` の中身を確認
2. **Undefined control sequence** — pandoc が生成した `\hypertarget` や `\tightlist` が残っている。定義を追加するか除去
3. **Missing $ inserted** — 本文中の `_` や `^` が数式モードの外にある。`\_` にエスケープするか `$...$` で囲む
4. **Package not found** — 不要なパッケージを除去

## 修正のルール

- **本文の内容は変えない。** LaTeX の構文だけを直す
- section 番号は elsarticle が自動で付けるので、本文中の「Section 3」等の相互参照はそのまま
- 数式（$$...$$）は pandoc が `\[...\]` に変換しているはず。`equation` 環境にはしない（tag が付いているものは `\tag{}` を使う）
- `\begin{thebibliography}` はそのまま使う。BibTeX には変換しない
- Figure 2 の挿入箇所（§4.6 の直前か直後）に `\includegraphics` を追加：
  ```latex
  \begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figure2_phase_diagram.pdf}
  \caption{Institutional feasibility by creditworthiness and self-sufficiency. Each cell is one of 25 blocs. Colour shows the set of policy regimes that remain feasible over 30 years. Under automation (b), the feasible frontier shifts inward and the regimes separate.}
  \label{fig:phase-diagram}
  \end{figure}
  ```

## 出力

- `papers/fugitive/main.pdf` — コンパイル済み PDF
- エラーがあった場合は修正した `main.tex` を上書き保存
