/* Investigation subgraph renderer shared by the static dashboard and the live investigation page.
   Parallel transfers between the same two accounts are merged into one edge (count + total) so dense
   neighbourhoods stay readable, and nodes are placed with a deterministic "flow" layout:
   sender side on the left, receiver side on the right, 2-hop accounts fanned out behind their 1-hop neighbour. */
(function(){
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const CUR = {'US Dollar':'USD','Euro':'EUR','Yuan':'CNY','Rupee':'INR','Yen':'JPY','Ruble':'RUB','Shekel':'ILS','UK Pound':'GBP',
    'Canadian Dollar':'CAD','Australian Dollar':'AUD','Swiss Franc':'CHF','Mexican Peso':'MXN','Brazil Real':'BRL','Saudi Riyal':'SAR','Bitcoin':'BTC'};
  const cur = c => CUR[c] || c;
  const compact = new Intl.NumberFormat('en-US', {notation:'compact', maximumFractionDigits:1});
  const full = new Intl.NumberFormat('en-US', {maximumFractionDigits:2});
  const ICON = {
    plus:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>',
    minus:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 12h14"/></svg>',
    fit:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4"/></svg>',
    tag:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12V4a1 1 0 0 1 1-1h8l9 9-9 9z"/><circle cx="7.5" cy="7.5" r="1.5"/></svg>',
    layout:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="6" cy="12" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="18" cy="18" r="2.5"/><path d="M8.3 11 15.7 7M8.3 13l7.4 4"/></svg>',
    png:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>'};

  function aggregate(g){
    const groups = new Map();
    for (const e of g.edges){
      const k = e.s + '→' + e.d;
      let x = groups.get(k);
      if (!x){ x = {id:'g' + groups.size, s:e.s, d:e.d, edges:[], hop:9, sums:{}}; groups.set(k, x); }
      x.edges.push(e); x.hop = Math.min(x.hop, e.hop);
      x.sums[e.cur] = (x.sums[e.cur] || 0) + (+e.amount || 0);
    }
    for (const x of groups.values()){
      x.edges.sort((a,b) => String(a.ts).localeCompare(String(b.ts)));
      const curs = Object.keys(x.sums), n = x.edges.length;
      const amt = curs.length === 1 ? `${compact.format(x.sums[curs[0]])} ${cur(curs[0])}` : `${curs.length} currencies`;
      x.label = n > 1 ? `${n}× · ${amt}` : amt;
    }
    return [...groups.values()];
  }

  // BFS distance from the centre accounts inside the subgraph
  function rings(nodes, groups, center){
    const adj = new Map(nodes.map(n => [n, new Set()]));
    groups.forEach(x => { if (x.s !== x.d){ adj.get(x.s).add(x.d); adj.get(x.d).add(x.s); } });
    const dist = new Map(center.map(c => [c, 0])), q = [...center];
    while (q.length){ const u = q.shift(); for (const v of adj.get(u) || []) if (!dist.has(v)){ dist.set(v, dist.get(u) + 1); q.push(v); } }
    nodes.forEach(n => dist.has(n) || dist.set(n, 3));
    return {adj, dist};
  }

  function flowLayout(nodes, groups, center){
    const {adj, dist} = rings(nodes, groups, center), pos = {}, parent = {};
    const C = [...new Set(center)];
    const rad = d => d * Math.PI / 180;
    const place = (cx, cy, list, mid, span, r) => list.forEach((n, i) => {
      const a = list.length === 1 ? mid : mid - span/2 + span * i / (list.length - 1);
      pos[n] = {x:cx + r*Math.cos(rad(a)), y:cy + r*Math.sin(rad(a))}; parent[n] = {x:cx, y:cy};
    });
    const r1 = nodes.filter(n => dist.get(n) === 1).sort();
    if (C.length === 1){
      pos[C[0]] = {x:0, y:0};
      const n = r1.length, r = Math.max(140, n * 13);
      place(0, 0, r1, -90 + (n > 1 ? 180 : 0), n > 1 ? 360 - 360/n : 0, r);
    } else {
      pos[C[0]] = {x:-100, y:0}; pos[C[1]] = {x:100, y:0};
      C.slice(2).forEach((c, i) => pos[c] = {x:0, y:(i + 1) * 90});
      const L = [], R = [], B = [];
      r1.forEach(n => { const a = adj.get(n), l = a.has(C[0]), r = a.has(C[1]); (l && r ? B : l ? L : R).push(n); });
      const arc = (list, cx, mid) => { const n = list.length, span = Math.min(230, Math.max(0, (n - 1) * 26)); place(cx, 0, list, mid, span, Math.max(140, n * 16)); };
      arc(L, -100, 180); arc(R, 100, 0);
      B.forEach((n, i) => { pos[n] = {x:0, y:(i % 2 ? 1 : -1) * (120 + Math.floor(i/2) * 56)}; parent[n] = {x:0, y:0}; });
    }
    for (let d = 2; d <= 3; d++){
      const kids = new Map();
      nodes.filter(n => dist.get(n) === d).sort().forEach(n => {
        const p = [...adj.get(n)].find(v => pos[v] && dist.get(v) < d) || C[0];
        if (!kids.has(p)) kids.set(p, []);
        kids.get(p).push(n);
      });
      for (const [p, list] of kids){
        const pp = pos[p] || {x:0, y:0}, anchor = parent[p] || {x:0, y:0};
        let dx = pp.x - anchor.x, dy = pp.y - anchor.y; const len = Math.hypot(dx, dy) || 1; dx /= len; dy /= len;
        const k = list.length, step = Math.min(30, 150 / k), base = Math.atan2(dy, dx) * 180 / Math.PI;
        list.forEach((n, j) => {
          const a = rad(base + (j - (k - 1)/2) * step), r = 100 + (k > 5 ? (j % 2) * 40 : 0);
          pos[n] = {x:pp.x + r*Math.cos(a), y:pp.y + r*Math.sin(a)}; parent[n] = pp;
        });
      }
    }
    return {pos, dist};
  }

  function create(wrap, opts = {}){
    wrap.classList.add('graph-wrap');
    wrap.innerHTML = `<div class="cy"></div><div class="graph-empty">${esc(opts.empty || 'Select a transaction to load its network.')}</div>
      <div class="graph-tools">
        <button class="btn" data-a="in" title="Zoom in">${ICON.plus}</button><button class="btn" data-a="out" title="Zoom out">${ICON.minus}</button>
        <button class="btn" data-a="fit" title="Fit to view">${ICON.fit}</button><button class="btn" data-a="labels" title="Labels: everything / key accounts and edges / focus only">${ICON.tag}</button>
        <button class="btn" data-a="layout" title="Switch layout (flow / force)">${ICON.layout}</button><button class="btn" data-a="png" title="Download high-resolution PNG">${ICON.png}</button>
      </div>
      <div class="graph-legend"><span><i class="dot" style="background:var(--risk)"></i>Focus account</span><span><i style="background:var(--risk)"></i>Selected transfer</span>
        <span><i style="background:var(--brand)"></i>1 hop</span><span><i style="background:var(--hop2)"></i>2 hops</span><span class="faint">n× = merged parallel transfers</span></div>
      <div class="graph-tip"></div>`;
    const box = wrap.querySelector('.cy'), tip = wrap.querySelector('.graph-tip'), empty = wrap.querySelector('.graph-empty');
    let cy = null, g = null, labelMode = 1, layoutMode = 'flow';

    function style(){
      const ink = css('--ink'), ink2 = css('--ink-2'), muted = css('--muted'), panel = css('--panel'), brand = css('--brand'), risk = css('--risk'), hop2 = css('--hop2');
      const font = css('--mono') || 'monospace', sans = css('--font') || 'sans-serif';
      return [
        {selector:'node', style:{'background-color':hop2, width:'data(size)', height:'data(size)', 'border-width':2, 'border-color':panel,
          'font-family':sans, 'font-size':11, color:ink2, 'text-valign':'bottom', 'text-margin-y':5,
          'text-outline-color':panel, 'text-outline-width':3, 'text-outline-opacity':1, 'min-zoomed-font-size':7, 'transition-property':'opacity', 'transition-duration':'120ms'}},
        {selector:'node[ring = 1]', style:{'background-color':brand}},
        {selector:'node[ring = 0]', style:{'background-color':risk, 'font-size':12, 'font-weight':600, color:ink, 'border-width':3,
          'underlay-color':risk, 'underlay-opacity':.18, 'underlay-padding':7, 'underlay-shape':'ellipse', 'text-margin-y':8}},
        {selector:'edge', style:{width:'data(w)', 'line-color':hop2, 'target-arrow-color':hop2, 'target-arrow-shape':'triangle', 'arrow-scale':.85,
          'curve-style':'bezier', 'control-point-step-size':46, 'loop-direction':'-50deg', 'loop-sweep':'-70deg', opacity:.9,
          'font-family':font, 'font-size':9.5, color:muted, 'text-rotation':'autorotate', 'text-margin-y':-7,
          'text-outline-color':panel, 'text-outline-width':2.5, 'min-zoomed-font-size':7, 'transition-property':'opacity', 'transition-duration':'120ms'}},
        {selector:'edge[hop = 1]', style:{'line-color':brand, 'target-arrow-color':brand}},
        {selector:'edge[hop = 0]', style:{'line-color':risk, 'target-arrow-color':risk, color:risk, 'font-weight':600, 'z-index':10}},
        {selector:'edge.lbl', style:{label:'data(label)'}},
        {selector:'node.nl, node.hl', style:{label:'data(name)'}},
        {selector:'.faded', style:{opacity:.12}},
        {selector:'edge.hl', style:{label:'data(label)', opacity:1, color:ink, 'z-index':20}},
        {selector:'node.hl', style:{color:ink, 'z-index':20}},
        {selector:'node:selected', style:{'border-color':brand, 'border-width':3}},
        {selector:'edge:selected', style:{'line-color':ink, 'target-arrow-color':ink, label:'data(label)', color:ink}}];
    }
    function applyLabels(){
      if (!cy) return;
      cy.edges().removeClass('lbl'); cy.nodes().removeClass('nl');
      // 2: every label; 1 (default): focus and 1-hop accounts plus key edges; 0: focus accounts only
      cy.nodes().filter(n => n.data('ring') <= [0, 1, 2][labelMode]).addClass('nl');
      if (labelMode === 2) cy.edges().addClass('lbl');
      else if (labelMode === 1) cy.edges().filter(e => e.data('hop') <= 1 || cy.edges().length <= 30).addClass('lbl');
    }
    function runLayout(fit = true){
      if (!cy) return;
      const opt = layoutMode === 'flow' ? {name:'preset', positions:n => g._pos[n.id()], padding:44, fit}
        : {name:'cose', animate:false, randomize:false, nodeRepulsion:() => 14000, idealEdgeLength:() => 110, edgeElasticity:() => 80, gravity:.25, numIter:1500, padding:44, fit};
      cy.layout(opt).run();
    }
    function showTip(html, ev){
      tip.innerHTML = html; tip.classList.add('on');
      const r = wrap.getBoundingClientRect(), p = ev.renderedPosition || ev.target.renderedPosition();
      let x = p.x + 14, y = p.y + 14;
      if (x + tip.offsetWidth > r.width - 8) x = p.x - tip.offsetWidth - 14;
      if (y + tip.offsetHeight > r.height - 8) y = p.y - tip.offsetHeight - 14;
      tip.style.left = Math.max(8, x) + 'px'; tip.style.top = Math.max(8, y) + 'px';
    }
    const hideTip = () => tip.classList.remove('on');
    const row = (a, b) => `<div class="r"><span>${esc(a)}</span><span>${esc(b)}</span></div>`;

    function draw(sub){
      g = sub; hideTip();
      if (cy){ cy.destroy(); cy = null; }
      if (!g || !g.nodes.length){ empty.style.display = 'grid'; empty.textContent = opts.empty || 'No transfers in this window.'; return; }
      empty.style.display = 'none';
      const groups = aggregate(g), {pos, dist} = flowLayout(g.nodes, groups, g.center);
      g._pos = pos; g._groups = groups;
      const deg = {}; groups.forEach(x => { deg[x.s] = (deg[x.s] || 0) + x.edges.length; deg[x.d] = (deg[x.d] || 0) + x.edges.length; });
      const acc = g.accounts || opts.accounts || {};
      const els = g.nodes.map(n => ({data:{id:n, name:acc[n]?.name || n, ring:Math.min(dist.get(n), 2), size:dist.get(n) === 0 ? 30 : 12 + Math.min(12, (deg[n] || 1) * 1.4)}}))
        .concat(groups.map(x => ({data:{id:x.id, source:x.s, target:x.d, hop:Math.min(x.hop, 2), label:x.label, w:1.3 + Math.log2(x.edges.length) * 1.4 + (x.hop === 0 ? 1.6 : 0)}})));
      cy = cytoscape({container:box, elements:els, style:style(), layout:{name:'preset', positions:n => pos[n.id()], padding:44},
        minZoom:.15, maxZoom:4, wheelSensitivity:.25, pixelRatio:Math.max(2, window.devicePixelRatio || 1), textureOnViewport:false, boxSelectionEnabled:false});
      if (layoutMode !== 'flow') runLayout();
      applyLabels();
      cy.on('mouseover', 'node', ev => {
        const n = ev.target, nb = n.closedNeighborhood();
        cy.elements().not(nb).addClass('faded'); n.connectedEdges().addClass('hl'); nb.nodes().addClass('hl');
        const out = groups.filter(x => x.s === n.id()).reduce((a, x) => a + x.edges.length, 0), inn = groups.filter(x => x.d === n.id()).reduce((a, x) => a + x.edges.length, 0);
        const a = acc[n.id()] || {};
        showTip(`<div class="t" style="font-family:var(--font)">${esc(a.name || n.id())}</div>${row('Account', n.id())}${a.bank ? row('Bank', a.bank) : ''}${row('Ring', ['focus', '1 hop', '2+ hops'][n.data('ring')])}${row('Outgoing transfers', out)}${row('Incoming transfers', inn)}`, ev);
      });
      cy.on('mouseover', 'edge', ev => {
        const x = groups.find(y => y.id === ev.target.id()); ev.target.addClass('hl'); ev.target.connectedNodes().addClass('hl');
        const lines = x.edges.slice(0, 6).map(e => row(String(e.ts).replace('T', ' ').slice(5, 16), `${full.format(e.amount)} ${cur(e.cur)}`)).join('');
        showTip(`<div class="t" style="font-family:var(--font)">${esc(acc[x.s]?.name || x.s)} → ${esc(acc[x.d]?.name || x.d)}</div>${lines}${x.edges.length > 6 ? `<div class="faint">+${x.edges.length - 6} more</div>` : ''}`, ev);
      });
      cy.on('mouseout', () => { cy.elements().removeClass('faded hl'); hideTip(); });
      cy.on('tap', 'edge', ev => { hideTip(); opts.onEdge && opts.onEdge(groups.find(y => y.id === ev.target.id()), g); });
      cy.on('tap', 'node', ev => { hideTip(); opts.onNode && opts.onNode(ev.target.id(), g); });
      const own = groups.find(x => x.hop === 0);
      if (own && opts.onEdge && opts.autoSelectOwn !== false) opts.onEdge(own, g, true);
    }

    wrap.querySelector('.graph-tools').addEventListener('click', ev => {
      const b = ev.target.closest('button'); if (!b || !cy) return;
      const a = b.dataset.a, c = {x:box.clientWidth/2, y:box.clientHeight/2};
      if (a === 'in') cy.zoom({level:cy.zoom() * 1.3, renderedPosition:c});
      if (a === 'out') cy.zoom({level:cy.zoom() / 1.3, renderedPosition:c});
      if (a === 'fit') cy.fit(undefined, 44);
      if (a === 'labels'){ labelMode = (labelMode + 1) % 3; applyLabels(); }
      if (a === 'layout'){ layoutMode = layoutMode === 'flow' ? 'force' : 'flow'; runLayout(); }
      if (a === 'png'){
        const url = cy.png({full:true, scale:3, bg:css('--panel')}), link = document.createElement('a');
        link.href = url; link.download = `subgraph_${(g.transaction_id || g.center[0]).replace(/[^\w-]/g, '_')}.png`; link.click();
      }
    });
    return {draw, resize(){ if (cy){ cy.resize(); cy.fit(undefined, 44); } }, restyle(){ if (cy){ cy.style(style()); applyLabels(); } }, get cy(){ return cy; }};
  }
  window.AMLGraph = {create, cur, esc};
})();
