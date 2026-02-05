/* global marked */

function escapeHtml(raw) {
  return String(raw)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function cleanMarkdownText(text) {
  return String(text)
    // Strip known dirty tag
    .replace(/TAGS_SHORT_READ_WARNING\s+(true|false)\s*/g, "")
    // 1) remove leading whitespace before headings
    .replace(/^\s+(#{1,6})/gm, "$1")
    // 2) normalize heading spaces (including full-width and nbsp)
    .replace(/(#{1,6})[\s\u3000\u00A0]+/gm, "$1 ")
    // 3) fix bold immediately after heading marker
    .replace(/(#{1,6} \*\*)[\s\u3000\u00A0]+/gm, "$1");
}

/**
 * 流式加载 Markdown 内容
 * @param {string} url - API 地址
 * @param {string} targetId - 内容显示容器 ID
 * @param {string} loadingId - 加载动画容器 ID
 * @param {object|null} payload - 可选：POST 的 JSON 数据；若提供则使用 POST
 */
async function loadStream(url, targetId, loadingId, payload = null) {
  const target = document.getElementById(targetId);
  const loading = document.getElementById(loadingId);
  if (!target) return;

  const showError = (msg) => {
    if (loading) {
      loading.style.display = "";
      loading.innerHTML = `<span style="color:#ef4444;font-weight:600">生成失败: ${escapeHtml(msg)}</span>`;
    }
  };

  const fetchOptions = payload
    ? {
        method: "POST",
        headers: { Accept: "text/plain", "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    : { method: "GET", headers: { Accept: "text/plain" } };

  try {
    const response = await fetch(url, fetchOptions);
    if (!response.ok) {
      const t = await response.text();
      throw new Error(`${response.status} ${t || response.statusText}`);
    }
    if (!response.body) {
      throw new Error("Streaming not supported in this browser");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let hasShown = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      if (!chunk) continue;

      buffer += chunk;

      if (!hasShown) {
        hasShown = true;
        if (loading) loading.style.display = "none";
        target.classList.remove("hidden");
        target.style.display = "";
      }

      const cleanText = cleanMarkdownText(buffer);
      const safeText = escapeHtml(cleanText);

      if (typeof marked !== "undefined") {
        target.innerHTML = marked.parse(safeText);
      } else {
        target.innerHTML = safeText.replace(/\n/g, "<br/>");
      }
    }
  } catch (err) {
    console.error("Stream error:", err);
    showError(err && err.message ? err.message : String(err));
  }
}

window.loadStream = loadStream;
