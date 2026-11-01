/* VLA Research Hub · 交互脚本
   1 主题切换  2 全站搜索(Ctrl+K)  3 列表筛选排序  4 TOC 高亮  5 进度条 */
(function () {
  "use strict";
  var BASE = window.__BASE || "";           // 到站点根的相对前缀，如 "../../"
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* ---------- 1. 主题 ---------- */
  var KEY = "vla-theme";
  var ICO_SUN = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  var ICO_MOON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    var b = $("#themebtn"); if (b) b.innerHTML = t === "dark" ? ICO_SUN : ICO_MOON;
    try { localStorage.setItem(KEY, t); } catch (e) {}
  }
  var saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) {}
  applyTheme(saved || (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  var tb = $("#themebtn");
  if (tb) tb.addEventListener("click", function () {
    applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });

  /* ---------- 2. 移动菜单 ---------- */
  var mb = $("#menubtn"), links = $("nav.top .links");
  if (mb && links) mb.addEventListener("click", function () { links.classList.toggle("open"); });

  /* ---------- 3. 搜索 ---------- */
  var overlay = $("#overlay"), input = $("#searchinput"), results = $("#results");
  var index = null, sel = -1, loading = false;

  function openSearch() {
    if (!overlay) return;
    overlay.classList.add("on");
    if (input) { input.value = ""; input.focus(); }
    if (results) results.innerHTML = '<div class="none">输入关键词：论文名、机构、技术词（如 flow matching、LIBERO、harness）</div>';
    document.body.style.overflow = "hidden";
    if (!index && !loading) loadIndex();
  }
  function closeSearch() {
    if (!overlay) return;
    overlay.classList.remove("on");
    document.body.style.overflow = "";
  }
  function loadIndex() {
    loading = true;
    fetch(BASE + "data/search.json").then(function (r) { return r.json(); })
      .then(function (d) { index = d; loading = false; if (input && input.value) run(input.value); })
      .catch(function () { results.innerHTML = '<div class="none">搜索索引加载失败</div>'; loading = false; });
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]; }); }
  function hl(text, q) {
    text = esc(text);
    if (!q) return text;
    try { return text.replace(new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi"), "<mark>$1</mark>"); }
    catch (e) { return text; }
  }
  function run(q) {
    q = (q || "").trim();
    if (!index) return;
    if (!q) { results.innerHTML = '<div class="none">输入关键词开始搜索（共 ' + index.length + ' 篇）</div>'; return; }
    var ql = q.toLowerCase(), hits = [];
    index.forEach(function (it) {
      var title = (it.t || "").toLowerCase(), sum = (it.s || "").toLowerCase(), tags = (it.k || "").toLowerCase();
      var sc = 0;
      if (title.indexOf(ql) >= 0) sc += 100;
      if (tags.indexOf(ql) >= 0) sc += 40;
      if (sum.indexOf(ql) >= 0) sc += 20;
      if (sc) hits.push({ it: it, sc: sc });
    });
    hits.sort(function (a, b) { return b.sc - a.sc; });
    hits = hits.slice(0, 24);
    sel = -1;
    if (!hits.length) { results.innerHTML = '<div class="none">没有匹配「' + esc(q) + '」的条目</div>'; return; }
    results.innerHTML = hits.map(function (h, i) {
      var it = h.it;
      return '<a href="' + BASE + it.u + '" data-i="' + i + '"><div class="rt">' + hl(it.t, q) +
        '</div><div class="rm"><span>' + esc(it.c || "") + '</span><span>·</span><span>' + esc(it.k || "") + "</span></div></a>";
    }).join("");
  }
  function move(d) {
    var as = $$("#results a"); if (!as.length) return;
    sel = (sel + d + as.length) % as.length;
    as.forEach(function (a, i) { a.classList.toggle("sel", i === sel); });
    as[sel].scrollIntoView({ block: "nearest" });
  }
  if (overlay) {
    overlay.addEventListener("click", function (e) { if (e.target === overlay) closeSearch(); });
    if (input) input.addEventListener("input", function () { run(input.value); });
    document.addEventListener("keydown", function (e) {
      var meta = e.metaKey || e.ctrlKey;
      if (meta && e.key.toLowerCase() === "k") { e.preventDefault(); openSearch(); return; }
      if (e.key === "Escape") { closeSearch(); return; }
      if (!overlay.classList.contains("on")) return;
      if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (e.key === "Enter") {
        var as = $$("#results a");
        if (as.length) { e.preventDefault(); (as[sel >= 0 ? sel : 0]).click(); }
      }
    });
    $$("[data-search]").forEach(function (b) { b.addEventListener("click", openSearch); });
  }

  /* ---------- 4. 列表页筛选 / 排序 ---------- */
  var list = $("#list");
  if (list) {
    var cards = $$(".pcard[data-cat]", list);
    var f = $("#filter"), sortSel = $("#sortsel");
    var activeCat = "all", kw = "";

    function apply() {
      var shown = 0;
      cards.forEach(function (c) {
        var okCat = activeCat === "all" || (c.dataset.cat || "").split(" ").indexOf(activeCat) >= 0;
        var hay = (c.dataset.hay || "").toLowerCase();
        var okKw = !kw || hay.indexOf(kw) >= 0;
        var show = okCat && okKw;
        c.style.display = show ? "" : "none";
        if (show) shown++;
      });
      $$(".group", list).forEach(function (g) {
        var any = $$(".pcard", g).some(function (c) { return c.style.display !== "none"; });
        g.style.display = any ? "" : "none";
      });
      var e = $("#empty"); if (e) e.style.display = shown ? "none" : "";
      var cn = $("#showncount"); if (cn) cn.textContent = shown;
    }
    function doSort() {
      if (!sortSel) return;
      var mode = sortSel.value;
      $$(".group", list).forEach(function (g) {
        var box = $(".cards", g); if (!box) return;
        var arr = $$(".pcard", box);
        arr.sort(function (a, b) {
          if (mode === "default") return (parseInt(a.dataset.idx || 0, 10) - parseInt(b.dataset.idx || 0, 10));
          if (mode === "name") return (a.dataset.name || "").localeCompare(b.dataset.name || "", "zh");
          if (mode === "name-d") return (b.dataset.name || "").localeCompare(a.dataset.name || "", "zh");
          if (mode === "year") return (parseInt(b.dataset.year || 0, 10) - parseInt(a.dataset.year || 0, 10));
          return 0;
        });
        arr.forEach(function (c) { box.appendChild(c); });
      });
    }
    if (f) f.addEventListener("input", function () { kw = f.value.trim().toLowerCase(); apply(); });
    $$(".chip[data-cat]").forEach(function (ch) {
      ch.addEventListener("click", function () {
        $$(".chip[data-cat]").forEach(function (o) { o.classList.remove("on"); });
        ch.classList.add("on");
        activeCat = ch.dataset.cat;
        apply();
      });
    });
    if (sortSel) sortSel.addEventListener("change", doSort);
    apply();
  }

  /* ---------- 5. TOC 高亮 ---------- */
  var toc = $(".toc");
  if (toc) {
    var heads = $$(".prose h2[id], .prose h3[id]");
    var linksT = $$(".toc a");
    function spy() {
      var y = window.scrollY + 96, cur = null;
      heads.forEach(function (h) { if (h.offsetTop <= y) cur = h.id; });
      linksT.forEach(function (a) {
        a.classList.toggle("on", a.getAttribute("href") === "#" + cur);
      });
    }
    window.addEventListener("scroll", spy, { passive: true });
    spy();
  }

  /* ---------- 6. 进度条 + 回到顶部 ---------- */
  var bar = $("#progress"), top = $(".totop");
  function onScroll() {
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    var p = max > 0 ? (h.scrollTop / max) * 100 : 0;
    if (bar) bar.style.width = p + "%";
    if (top) top.classList.toggle("show", h.scrollTop > 700);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  if (top) top.addEventListener("click", function () { window.scrollTo({ top: 0, behavior: "smooth" }); });
})();
