---
name: Bug report
about: A crash, an error or a wrong answer - with the output of ./abyss.sh --check and --report from your PC
title: "Bug: <what happened> (<model> on <your GPU>)"
labels: bug
---

<!-- Before you post: stop the model server, then run
       ./abyss.sh --check          (Windows: START-ABYSS.bat --check; ~15-30 min, --quick ~5 min)
       ./abyss.sh --report         (Windows: START-ABYSS.bat --report)
     and attach abyss-report.txt (it includes abyss-check.txt and abyss-bench.txt when they exist).
     These files have no host name, user name or folder path, and nothing is sent anywhere; read them before you post.
     docs/BENCH.md explains each line. -->

**What happened** (the error, the crash, the wrong answer - the message or a screenshot):

**Steps to reproduce** (what you asked, which client - the dashboard, Claude Code, OpenCode, the API - and how long the
prompt or the conversation was):

1.
2.

**What you expected**:

**Which files** (the GGUF folder, e.g. `zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF`, and the quant):

**abyss-report.txt** (attach it, or paste it inside a ``` block):

**Did ./abyss.sh --check pass?** (PASS / WARN / FAIL, and the step that failed):
