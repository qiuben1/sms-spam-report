# -*- coding: utf-8 -*-
# 训练+生成报告到 site/ → 推送到 GitHub Pages → 生成永久二维码（包含源码展示/下载）
import os, re, string, math, html as html_lib, subprocess, sys, shutil
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import qrcode
except Exception:
    print('缺少依赖：请先执行  pip install "qrcode[pil]"')
    sys.exit(1)

# ========= 配置（按需修改） =========
REPO_URL = "https://github.com/qiuben1/sms-spam-report.git"
BRANCH   = "main"
OUT_DIR  = "site"
# ==================================

# ---------- 数据 ----------
def load_stopwords(path="stopwords.txt"):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return set(w.strip() for w in f if w.strip())
stopwords = load_stopwords("stopwords.txt")
data = pd.read_csv("SMSSpamCollection.txt", sep="\t", header=None, names=["labels", "messages"])
X, y = data["messages"], data["labels"]
print("数据前5行：\n", data.head())

# ---------- 预处理 ----------
_punc_re = re.compile(f'[{re.escape(string.punctuation)}]')
def preprocess(text: str):
    text = text.lower()
    text = _punc_re.sub(" ", text)
    return [w for w in text.split() if w not in stopwords]

# ---------- 朴素贝叶斯 ----------
class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.vocabulary = set()
        self.class_total = defaultdict(int)
        self.word_total  = defaultdict(int)
        self.word_given_class = defaultdict(lambda: defaultdict(int))
        self.classes_, self.vocab_size, self.log_prior_ = [], 0, {}

    def fit(self, X_iter, y_iter):
        for text, label in zip(X_iter, y_iter):
            words = preprocess(text)
            self.class_total[label] += 1
            for w in words:
                self.vocabulary.add(w)
                self.word_given_class[label][w] += 1
                self.word_total[label] += 1
        self.classes_ = list(self.class_total.keys())
        self.vocab_size = len(self.vocabulary)
        N = sum(self.class_total.values())
        self.log_prior_ = {c: math.log(self.class_total[c] / N) for c in self.classes_}
        return self

    def _log_like_sum(self, tokens, c: str) -> float:
        denom = self.word_total[c] + self.alpha * self.vocab_size
        s = 0.0
        for w in tokens:
            num = self.word_given_class[c].get(w, 0) + self.alpha
            s += math.log(num / denom)
        return s

    def predict_log_proba(self, X_iter):
        if isinstance(X_iter, str): X_iter = [X_iter]
        out = []
        for text in X_iter:
            toks = preprocess(text)
            scores = [self.log_prior_[c] + self._log_like_sum(toks, c) for c in self.classes_]
            m = max(scores); exps = [math.exp(s - m) for s in scores]; Z = sum(exps)
            out.append([math.log(e / Z) for e in exps])
        return np.array(out)

    def predict_proba(self, X_iter):
        return np.exp(self.predict_log_proba(X_iter))

    def predict(self, X_iter):
        if isinstance(X_iter, str): X_iter = [X_iter]
        logp = self.predict_log_proba(X_iter)
        idx = logp.argmax(axis=1)
        return np.array([self.classes_[i] for i in idx])

# ---------- 训练评估 ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
clf = NaiveBayesClassifier(alpha=1.0).fit(X_train, y_train)
y_pred = clf.predict(X_test)

label_order = ["ham","spam"]
y_true = np.asarray(y_test); y_pred = np.asarray(y_pred)
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
rec  = recall_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
f1   = f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
cm   = confusion_matrix(y_true, y_pred, labels=label_order)

print(f"\nAccuracy: {acc:.4f}")
print(f"Precision(macro): {prec:.4f}  Recall(macro): {rec:.4f}  F1(macro): {f1:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n",
      classification_report(y_true, y_pred, labels=label_order, target_names=label_order, digits=4, zero_division=0))

# ---------- 10 段文本 ----------
messages = [
    'how are you .fine, thank you and you?',
    "Good News, You've won a big prize, please call 00861888888888 for more information",
    'are you ok',
    'not at all',
    'WINNER, You have won a 1 week FREE membership in our £100,000 Prize Jackpot!',
    'winner winner chicken dinner',
    'Congratulations on your invitation to join Honor Society!',
    'I love you guys.',
    'Extra large discount, three bamboo rats and three 10 yuan',
    'good good study day day up'
]
pred_ms  = clf.predict(messages)
proba_ms = clf.predict_proba(messages)
cls2idx  = {c:i for i,c in enumerate(clf.classes_)}
p_ham, p_spam = proba_ms[:, cls2idx["ham"]], proba_ms[:, cls2idx["spam"]]

df_out = pd.DataFrame({"id":range(1,len(messages)+1),
                       "message":messages,
                       "prediction":pred_ms,
                       "P_ham":p_ham, "P_spam":p_spam})

# ---------- 生成静态站点 ----------
os.makedirs(OUT_DIR, exist_ok=True)

def save_cm_img(cm, labels, path_png):
    plt.figure(figsize=(4.8,4.2))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(labels)), labels); plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,int(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.savefig(path_png, dpi=200, bbox_inches="tight"); plt.close()

def save_counts_bar(preds, path_png):
    counts = pd.Series(preds).value_counts().reindex(["ham","spam"]).fillna(0)
    plt.figure(figsize=(4.8,3.0))
    plt.bar(counts.index, counts.values)
    for x,v in zip(counts.index, counts.values):
        plt.text(x, v+0.05, int(v), ha="center")
    plt.title("Predicted Class Counts for 10 Messages"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(path_png, dpi=200, bbox_inches="tight"); plt.close()

save_cm_img(cm, label_order, os.path.join(OUT_DIR,"confusion_matrix.png"))
save_counts_bar(pred_ms, os.path.join(OUT_DIR,"messages_pred_counts.png"))
df_out.to_csv(os.path.join(OUT_DIR,"messages_predictions.csv"), index=False, encoding="utf-8")

# === 新增：读取并复制源码到 site/，并在页面展示 ===
SRC_PATH = os.path.abspath(__file__) if "__file__" in globals() else None
try:
    code_text = open(SRC_PATH, "r", encoding="utf-8", errors="ignore").read()
except Exception:
    code_text, SRC_PATH = "# 源代码读取失败", None

code_html = html_lib.escape(code_text)  # 转义后放 <pre> 里
download_name = None
if SRC_PATH:
    try:
        download_name = os.path.basename(SRC_PATH)
        shutil.copy2(SRC_PATH, os.path.join(OUT_DIR, download_name))
    except Exception as e:
        print("[WARN] 复制源码到 site/ 失败：", e)
        download_name = None

html = f"""<!doctype html>
<meta charset="utf-8">
<title>SMS Spam Naive Bayes - Report</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin: 20px; }}
 h1,h2 {{ margin: 14px 0; }}
 table {{ border-collapse: collapse; width:100%; }} th,td {{ border:1px solid #ddd; padding:6px 8px; }} th{{background:#fafafa}}
 img {{ max-width:100%; height:auto; }}
 pre {{ background:#f7f7f7; padding:12px; border-radius:8px; white-space:pre-wrap; word-break:break-word; }}
 details > summary {{ cursor:pointer; font-weight:600; }}
 .muted {{ color:#666; font-size:0.95em; }}
</style>
<h1>SMS Spam Naive Bayes - Report</h1>
<ul>
  <li>Accuracy: {acc:.4f}</li>
  <li>Precision (macro): {prec:.4f}</li>
  <li>Recall (macro): {rec:.4f}</li>
  <li>F1 (macro): {f1:.4f}</li>
</ul>
<h2>Confusion Matrix</h2>
<img src="confusion_matrix.png" alt="confusion matrix">
<h2>Predicted Class Counts (10 Messages)</h2>
<img src="messages_pred_counts.png" alt="pred counts">
<h2>Predictions for 10 Messages</h2>
{df_out.to_html(index=False, escape=True)}
<h2>Source Code</h2>
<p class="muted">下面展示的是本页面生成脚本的完整源码。</p>
{"<p><a href='" + download_name + "' download>⬇ 下载 publish_report.py</a></p>" if download_name else ""}
<details open>
  <summary>展开/收起 源码</summary>
  <pre>{code_html}</pre>
</details>
"""
open(os.path.join(OUT_DIR,"index.html"), "w", encoding="utf-8").write(html)
print("已生成静态站点：", os.path.abspath(OUT_DIR))

# ---------- 推送到 GitHub ----------
def run(cmd, cwd):
    env = os.environ.copy()
    env.update({"LC_ALL":"C.UTF-8","LANG":"C.UTF-8"})
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True,
                         encoding="utf-8", errors="ignore", env=env)
    if res.returncode != 0:
        print("命令失败:", " ".join(cmd))
        if res.stdout: print("STDOUT:\n", res.stdout)
        if res.stderr: print("STDERR:\n", res.stderr)
    return res

def publish(out_dir, repo_url, branch="main"):
    if not os.path.isdir(os.path.join(out_dir, ".git")):
        run(["git","init","-b",branch], out_dir)

    m = re.match(r"https://github\.com/([^/]+)/([^/.]+)(?:\.git)?", repo_url)
    username = m.group(1) if m else "pages-bot"
    run(["git","config","user.name", username], out_dir)
    run(["git","config","user.email", f"{username}@users.noreply.github.com"], out_dir)

    cur = run(["git","remote","get-url","origin"], out_dir)
    if cur.returncode != 0:
        run(["git","remote","add","origin", repo_url], out_dir)
    else:
        run(["git","remote","set-url","origin", repo_url], out_dir)

    ls = run(["git","ls-remote","--heads","origin", branch], out_dir)
    if ls.returncode == 0 and ls.stdout.strip():
        run(["git","fetch","origin", branch], out_dir)
        run(["git","checkout","-B", branch, f"origin/{branch}"], out_dir)
    else:
        run(["git","checkout","-B", branch], out_dir)

    run(["git","add","."], out_dir)
    cmt = run(["git","commit","-m","update report"], out_dir)
    if cmt.returncode != 0 and "nothing to commit" not in (cmt.stdout+cmt.stderr).lower():
        return False
    push = run(["git","push","-u","origin", branch], out_dir)
    return push.returncode == 0

ok = publish(OUT_DIR, REPO_URL, BRANCH)
if not ok:
    print("\n推送失败：请确认已登录 GitHub（或在提示时输入 PAT），以及 REPO_URL 正确。")
    sys.exit(1)

# ---------- 生成永久二维码（Pages 链接） ----------
def pages_url_from_repo(repo_url: str):
    m = re.match(r"https://github\.com/([^/]+)/([^/.]+)(?:\.git)?", repo_url)
    if not m: return None
    user, repo = m.groups()
    return f"https://{user}.github.io/{repo}/"

pages_url = pages_url_from_repo(REPO_URL)
qr_path = os.path.join(OUT_DIR, "permanent_qr.png")
qrcode.make(pages_url).save(qr_path)

print("\n✅ 已发布到 GitHub Pages：", pages_url)
print("✅ 永久二维码已生成：", os.path.abspath(qr_path))
print("（首次启用 Pages 可能延迟 1–2 分钟再刷新/扫码）")
