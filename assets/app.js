const CATS = ["冻结模型编排","世界模型与想象","编码 Agent 与系统进化","技能库与在线探索","评测与基准","知识与记忆","通用 Agent Harness","智能迁移与 ICL","综述与横评"];
let PAPERS = [], activeCat = new URLSearchParams(location.search).get("cat") || "全部";

async function boot(){
  const res = await fetch("data/papers.json");
  PAPERS = await res.json();
  document.getElementById("stat-papers").textContent = PAPERS.length;
  document.getElementById("stat-cats").textContent = new Set(PAPERS.map(p=>p.cat)).size;
  renderChips(); render(); timeline();
  document.getElementById("q").addEventListener("input", render);
}
function renderChips(){
  const box = document.getElementById("chips");
  const mk = (label,val)=>`<button class="chip ${activeCat===val?"on":""}" data-v="${val}">${label}</button>`;
  box.innerHTML = mk("全部","全部") + CATS.map(c=>mk(c,c)).join("");
  box.querySelectorAll("button").forEach(b=>b.onclick=()=>{activeCat=b.dataset.v;renderChips();render();});
}
function match(p,q){
  if(!q) return true;
  q=q.toLowerCase();
  return [p.title,p.oneliner,p.metric,p.org,p.arxiv,p.cat].join(" ").toLowerCase().includes(q);
}
function render(){
  const q = document.getElementById("q").value.trim();
  const list = PAPERS.filter(p=>(activeCat==="全部"||p.cat===activeCat)&&match(p,q));
  const grid = document.getElementById("grid");
  grid.innerHTML = list.map(p=>`
    <div class="card">
      ${p.figs?`<div class="thumb"><img loading="lazy" src="assets/figures/${p.slug}/fig-01.png" alt=""></div>`:""}
      <h3><a href="papers/${p.slug}.html">${p.title}</a></h3>
      <p class="one">${p.oneliner}</p>
      <p class="metric">⭐ ${p.metric}</p>
      <div class="foot">
        <span class="chip cat">${p.cat}</span>
        ${p.date?`<span class="chip">${p.date.slice(0,7)}</span>`:""}
        ${p.arxiv?`<a class="chip" target="_blank" href="https://arxiv.org/abs/${p.arxiv}">arXiv:${p.arxiv}</a>`:""}
        <a class="chip" href="papers/${p.slug}.html">阅读 →</a>
      </div>
    </div>`).join("");
  document.getElementById("empty").style.display = list.length ? "none":"block";
}
function timeline(){
  const el = document.getElementById("timeline");
  const items = PAPERS.filter(p=>p.date && p.cat!=="综述与横评").sort((a,b)=>a.date<b.date?-1:1);
  el.innerHTML = items.map(p=>`
    <div class="tl-item"><span class="d">${p.date}</span> · <a href="papers/${p.slug}.html">${p.title}</a>
    <span class="c"> — ${p.cat}</span></div>`).join("");
}
// if deployed, show repo link automatically (set window.GH_USER in index.html or infer from location)
const host = location.hostname;
if(host.endsWith(".github.io")){
  const user = host.split(".")[0];
  const repoGuess = location.pathname.split("/")[1];
  const gh = document.getElementById("gh-link");
  gh.href = `https://github.com/${user}/${repoGuess||""}`; gh.style.display="";
}
boot();
