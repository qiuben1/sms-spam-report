# -*- coding: utf-8 -*-  # 指定源码文件使用 UTF-8 编码，避免中文乱码
# 训练+生成报告到 site/ → 推送到 GitHub Pages → 生成永久二维码（包含源码展示/下载）

import os, re, string, math, html as html_lib, subprocess, sys, shutil  # 常用标准库；html_lib 用于把源码转义成可安全显示的 HTML
import numpy as np                    # 数值计算
import pandas as pd                   # 表格数据处理
from collections import defaultdict   # 计数字典，未命中自动为 0
from sklearn.model_selection import train_test_split  # 划分训练/测试集
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)  # 常用评估指标

import matplotlib
matplotlib.use("Agg")   # 使用无界面的绘图后端，避免 Windows 上缺少 GUI 时的 _tkinter/TclError
import matplotlib.pyplot as plt  # 画图

try:
    import qrcode  # 生成二维码
except Exception:
    print('缺少依赖：请先执行  pip install "qrcode[pil]"')  # 友好提示缺包
    sys.exit(1)  # 中止程序

# ========= 配置（按需修改） =========
REPO_URL = "https://github.com/qiuben1/sms-spam-report.git"  # 你的 GitHub 仓库地址（用于 Pages）
BRANCH   = "main"   # 要推送的分支；仓库设置里将 main 作为 Pages 来源
OUT_DIR  = "site"   # 生成的静态网站目录（会被推送）
# ==================================

# ---------- 数据 ----------
def load_stopwords(path="stopwords.txt"):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:  # 读取停用词，忽略非法字符
        return set(w.strip() for w in f if w.strip())              # 去空行，存成集合
stopwords = load_stopwords("stopwords.txt")  # 加载停用词
data = pd.read_csv("SMSSpamCollection.txt", sep="\t", header=None, names=["labels", "messages"])  # 读取短信数据集（制表符分隔）
X, y = data["messages"], data["labels"]  # 文本与标签
print("数据前5行：\n", data.head())  # 快速检查读取是否正确

# ---------- 预处理 ----------
_punc_re = re.compile(f'[{re.escape(string.punctuation)}]')  # 正则：匹配英文标点；re.escape 防止特殊字符被误解释
def preprocess(text: str):
    text = text.lower()              # 统一转小写
    text = _punc_re.sub(" ", text)   # 标点替换为空格
    return [w for w in text.split() if w not in stopwords]  # 分词+去停用词（返回词列表）

# ---------- 朴素贝叶斯 ----------
class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)                   # 拉普拉斯平滑系数
        self.vocabulary = set()                     # 全局词表
        self.class_total = defaultdict(int)         # 每个类别的文档数
        self.word_total  = defaultdict(int)         # 每个类别内的词频总和
        self.word_given_class = defaultdict(lambda: defaultdict(int))  # 类别 -> 词 -> 计数
        self.classes_, self.vocab_size, self.log_prior_ = [], 0, {}    # 类别列表、词表大小、先验对数

    def fit(self, X_iter, y_iter):
        for text, label in zip(X_iter, y_iter):     # 逐条样本
            words = preprocess(text)                # 预处理得到词列表
            self.class_total[label] += 1            # 该类文档数 +1（用于计算先验）
            for w in words:                         # 遍历文档中的词
                self.vocabulary.add(w)              # 加入全局词表
                self.word_given_class[label][w] += 1  # 该类下该词计数 +1
                self.word_total[label] += 1         # 该类所有词计数 +1
        self.classes_ = list(self.class_total.keys())  # 训练后得到类别集合（如 ["ham","spam"]）
        self.vocab_size = len(self.vocabulary)         # 词表大小 |V|
        N = sum(self.class_total.values())             # 总文档数
        self.log_prior_ = {c: math.log(self.class_total[c] / N) for c in self.classes_}  # 先验 P(c) 的对数
        return self

    def _log_like_sum(self, tokens, c: str) -> float:
        denom = self.word_total[c] + self.alpha * self.vocab_size  # 分母：该类词频总和 + α|V|
        s = 0.0
        for w in tokens:
            num = self.word_given_class[c].get(w, 0) + self.alpha  # 分子：该类中的词频 + α
            s += math.log(num / denom)                             # 使用对数避免下溢
        return s

    def predict_log_proba(self, X_iter):
        if isinstance(X_iter, str): X_iter = [X_iter]  # 兼容直接传入字符串
        out = []
        for text in X_iter:
            toks = preprocess(text)  # 词列表
            # 计算每个类别的后验对数 ~ logP(c) + Σ logP(w|c)
            scores = [self.log_prior_[c] + self._log_like_sum(toks, c) for c in self.classes_]
            # 做一个 log-sum-exp 规范化，得到 log 概率（数值更稳定）
            m = max(scores); exps = [math.exp(s - m) for s in scores]; Z = sum(exps)
            out.append([math.log(e / Z) for e in exps])
        return np.array(out)

    def predict_proba(self, X_iter):
        return np.exp(self.predict_log_proba(X_iter))  # 把对数概率还原为普通概率

    def predict(self, X_iter):
        if isinstance(X_iter, str): X_iter = [X_iter]
        logp = self.predict_log_proba(X_iter)  # 得到每个类别的 log 概率
        idx = logp.argmax(axis=1)              # 取最大者的索引
        return np.array([self.classes_[i] for i in idx])  # 映射回类别名

# ---------- 训练评估 ----------
X_train, X_test, y_train, y_test = train_test_split(  # 划分训练/测试集（按 y 分层，保持类别比例）
    X, y, stratify=y, test_size=0.2, random_state=42
)
clf = NaiveBayesClassifier(alpha=1.0).fit(X_train, y_train)  # 训练模型
y_pred = clf.predict(X_test)  # 在测试集上预测

label_order = ["ham","spam"]  # 固定评估时的类别顺序
y_true = np.asarray(y_test); y_pred = np.asarray(y_pred)  # 转为数组，避免对齐问题
acc  = accuracy_score(y_true, y_pred)  # 准确率
prec = precision_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)  # 宏平均精确率
rec  = recall_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)     # 宏平均召回率
f1   = f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)         # 宏平均 F1
cm   = confusion_matrix(y_true, y_pred, labels=label_order)  # 混淆矩阵

print(f"\nAccuracy: {acc:.4f}")  # 打印指标
print(f"Precision(macro): {prec:.4f}  Recall(macro): {rec:.4f}  F1(macro): {f1:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n",
      classification_report(y_true, y_pred, labels=label_order, target_names=label_order, digits=4, zero_division=0))

# ---------- 10 段文本 ----------
messages = [  # 作业要求：给定 10 段文本，判断是否垃圾
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
pred_ms  = clf.predict(messages)         # 每条文本的预测类别
proba_ms = clf.predict_proba(messages)   # 对应的后验概率
cls2idx  = {c:i for i,c in enumerate(clf.classes_)}  # 类名到列索引的映射
p_ham, p_spam = proba_ms[:, cls2idx["ham"]], proba_ms[:, cls2idx["spam"]]  # 提取两类的概率列

# 汇总为表格，后续直接渲染到 HTML
df_out = pd.DataFrame({"id":range(1,len(messages)+1),
                       "message":messages,
                       "prediction":pred_ms,
                       "P_ham":p_ham, "P_spam":p_spam})

# ---------- 生成静态站点 ----------
os.makedirs(OUT_DIR, exist_ok=True)  # 确保输出目录存在

def save_cm_img(cm, labels, path_png):
    plt.figure(figsize=(4.8,4.2))
    plt.imshow(cm, cmap="Blues")           # 热力图
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(labels)), labels); plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):           # 在格子上写入计数
        for j in range(cm.shape[1]):
            plt.text(j,i,int(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.savefig(path_png, dpi=200, bbox_inches="tight"); plt.close()  # 保存图片并关闭图窗

def save_counts_bar(preds, path_png):
    counts = pd.Series(preds).value_counts().reindex(["ham","spam"]).fillna(0)  # 统计 10 条预测的类别分布
    plt.figure(figsize=(4.8,3.0))
    plt.bar(counts.index, counts.values)  # 柱状图
    for x,v in zip(counts.index, counts.values):
        plt.text(x, v+0.05, int(v), ha="center")  # 在柱子上标注数值
    plt.title("Predicted Class Counts for 10 Messages"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(path_png, dpi=200, bbox_inches="tight"); plt.close()

save_cm_img(cm, label_order, os.path.join(OUT_DIR,"confusion_matrix.png"))          # 保存混淆矩阵图
save_counts_bar(pred_ms, os.path.join(OUT_DIR,"messages_pred_counts.png"))          # 保存 10 条分布图
df_out.to_csv(os.path.join(OUT_DIR,"messages_predictions.csv"), index=False, encoding="utf-8")  # 保存 CSV

# === 新增：读取并复制源码到 site/，并在页面展示 ===
SRC_PATH = os.path.abspath(__file__) if "__file__" in globals() else None  # 当前脚本的绝对路径
try:
    code_text = open(SRC_PATH, "r", encoding="utf-8", errors="ignore").read()  # 读入全部源码
except Exception:
    code_text, SRC_PATH = "# 源代码读取失败", None  # 容错：读取失败时给出提示

code_html = html_lib.escape(code_text)  # 将源码做 HTML 转义，防止标签被浏览器误解析
download_name = None
if SRC_PATH:
    try:
        download_name = os.path.basename(SRC_PATH)  # 生成下载文件名
        shutil.copy2(SRC_PATH, os.path.join(OUT_DIR, download_name))  # 把脚本原文件复制到 site/，供直接下载
    except Exception as e:
        print("[WARN] 复制源码到 site/ 失败：", e)
        download_name = None

# 下方为报告页面 HTML（用 f-string 拼接指标和表格）
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
open(os.path.join(OUT_DIR,"index.html"), "w", encoding="utf-8").write(html)  # 写出最终网页
print("已生成静态站点：", os.path.abspath(OUT_DIR))

# ---------- 推送到 GitHub ----------
def run(cmd, cwd):
    env = os.environ.copy()
    env.update({"LC_ALL":"C.UTF-8","LANG":"C.UTF-8"})  # 统一编码，防止 Windows 控制台乱码/解码失败
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True,
                         encoding="utf-8", errors="ignore", env=env)  # 捕获输出便于调试
    if res.returncode != 0:
        print("命令失败:", " ".join(cmd))         # 打印失败命令
        if res.stdout: print("STDOUT:\n", res.stdout)
        if res.stderr: print("STDERR:\n", res.stderr)
    return res

def publish(out_dir, repo_url, branch="main"):
    # 若没有 .git，初始化一个仓库并指定默认分支
    if not os.path.isdir(os.path.join(out_dir, ".git")):
        run(["git","init","-b",branch], out_dir)

    # 基本身份配置（用仓库 owner 生成一个 noreply 邮件）
    m = re.match(r"https://github\.com/([^/]+)/([^/.]+)(?:\.git)?", repo_url)
    username = m.group(1) if m else "pages-bot"
    run(["git","config","user.name", username], out_dir)
    run(["git","config","user.email", f"{username}@users.noreply.github.com"], out_dir)

    # 绑定（或更新）远端 origin
    cur = run(["git","remote","get-url","origin"], out_dir)
    if cur.returncode != 0:
        run(["git","remote","add","origin", repo_url], out_dir)
    else:
        run(["git","remote","set-url","origin", repo_url], out_dir)

    # 若远端已有 main，则基于远端检出，避免“不相关历史/冲突”
    ls = run(["git","ls-remote","--heads","origin", branch], out_dir)
    if ls.returncode == 0 and ls.stdout.strip():
        run(["git","fetch","origin", branch], out_dir)
        run(["git","checkout","-B", branch, f"origin/{branch}"], out_dir)
    else:
        run(["git","checkout","-B", branch], out_dir)

    # 提交并推送
    run(["git","add","."], out_dir)
    cmt = run(["git","commit","-m","update report"], out_dir)
    if cmt.returncode != 0 and "nothing to commit" not in (cmt.stdout+cmt.stderr).lower():
        return False  # 提交失败且不是“无改动”
    push = run(["git","push","-u","origin", branch], out_dir)
    return push.returncode == 0  # True=成功

ok = publish(OUT_DIR, REPO_URL, BRANCH)  # 执行发布
if not ok:
    print("\n推送失败：请确认已登录 GitHub（或在提示时输入 PAT），以及 REPO_URL 正确。")
    sys.exit(1)

# ---------- 生成永久二维码（Pages 链接） ----------
def pages_url_from_repo(repo_url: str):
    m = re.match(r"https://github\.com/([^/]+)/([^/.]+)(?:\.git)?", repo_url)  # 解析出用户名和仓库名
    if not m: return None
    user, repo = m.groups()
    return f"https://{user}.github.io/{repo}/"  # Pages 访问地址固定格式

pages_url = pages_url_from_repo(REPO_URL)  # 得到最终的公开链接
qr_path = os.path.join(OUT_DIR, "permanent_qr.png")  # 二维码输出路径
qrcode.make(pages_url).save(qr_path)  # 生成并保存二维码

print("\n✅ 已发布到 GitHub Pages：", pages_url)          # 终端提示
print("✅ 永久二维码已生成：", os.path.abspath(qr_path))
print("（首次启用 Pages 可能延迟 1–2 分钟再刷新/扫码）")     # Pages 初次生效可能有延迟
