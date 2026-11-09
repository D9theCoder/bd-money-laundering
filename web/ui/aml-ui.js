/* Styled dropdowns. Each <select> stays in the DOM (hidden) as the source of truth, so existing code that reads
   .value, sets .value / .length, adds options or listens for "change" keeps working; a button + menu is drawn on top. */
(function(){
  const CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 5 5 9-10"/></svg>';
  const CHEV = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="m7 10 5 5 5-5"/></svg>';
  let open = null;

  function close(){
    if (!open) return;
    open.menu.remove(); open.btn.setAttribute('aria-expanded', 'false'); open.wrap.classList.remove('open');
    open = null;
  }
  document.addEventListener('mousedown', e => { if (open && !open.menu.contains(e.target) && !open.wrap.contains(e.target)) close(); });
  window.addEventListener('resize', close);
  document.addEventListener('scroll', e => { if (open && !open.menu.contains(e.target)) close(); }, true);

  function enhance(sel){
    if (sel.dataset.xsel) return;
    sel.dataset.xsel = '1';
    const wrap = document.createElement('div'), btn = document.createElement('button');
    wrap.className = 'xsel ' + sel.className;
    wrap.style.cssText = sel.style.cssText;
    if (sel.closest('.pager')) wrap.classList.add('sm');
    btn.type = 'button'; btn.className = 'xsel-btn'; btn.title = sel.title || '';
    btn.setAttribute('aria-haspopup', 'listbox'); btn.setAttribute('aria-expanded', 'false');
    btn.innerHTML = `<span class="xsel-val"></span>${CHEV}`;
    wrap.appendChild(btn);
    sel.after(wrap); sel.hidden = true; sel.tabIndex = -1;
    const label = sel.closest('label');
    if (label) label.addEventListener('click', e => { if (e.target === label || !wrap.contains(e.target)) e.preventDefault(); });

    const sync = () => { const o = sel.options[sel.selectedIndex]; btn.querySelector('.xsel-val').textContent = o ? o.text : ''; };
    for (const prop of ['value', 'selectedIndex', 'length']){
      const d = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, prop);
      Object.defineProperty(sel, prop, {configurable:true, get(){ return d.get.call(sel); }, set(v){ d.set.call(sel, v); sync(); }});
    }
    new MutationObserver(sync).observe(sel, {childList:true, subtree:true, characterData:true});
    sel.addEventListener('change', sync);
    sync();

    function pick(i){
      close();
      if (i === sel.selectedIndex) return;
      sel.selectedIndex = i;
      sel.dispatchEvent(new Event('input', {bubbles:true}));
      sel.dispatchEvent(new Event('change', {bubbles:true}));
      btn.focus();
    }
    function show(){
      if (open && open.sel === sel){ close(); return; }
      close();
      const menu = document.createElement('div');
      menu.className = 'xsel-menu'; menu.setAttribute('role', 'listbox');
      menu.innerHTML = [...sel.options].map((o, i) => `<div class="xsel-opt${i === sel.selectedIndex ? ' on' : ''}" role="option" data-i="${i}">
        <span class="xsel-check">${i === sel.selectedIndex ? CHECK : ''}</span><span>${o.text.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</span></div>`).join('');
      document.body.appendChild(menu);
      const r = btn.getBoundingClientRect(), h = Math.min(menu.scrollHeight, 320), below = innerHeight - r.bottom - 8;
      menu.style.minWidth = r.width + 'px';
      menu.style.left = Math.max(8, Math.min(r.left, innerWidth - menu.offsetWidth - 8)) + 'px';
      menu.style.top = (below >= h || below >= r.top ? r.bottom + 4 : r.top - h - 4) + 'px';
      menu.addEventListener('mousedown', e => e.preventDefault());
      menu.addEventListener('click', e => { const o = e.target.closest('.xsel-opt'); if (o) pick(+o.dataset.i); });
      menu.addEventListener('mousemove', e => { const o = e.target.closest('.xsel-opt'); if (o) focusOpt(+o.dataset.i, false); });
      open = {sel, btn, wrap, menu, active:sel.selectedIndex};
      btn.setAttribute('aria-expanded', 'true'); wrap.classList.add('open');
      focusOpt(Math.max(0, sel.selectedIndex), true);
    }
    function focusOpt(i, scroll){
      if (!open) return;
      const opts = open.menu.querySelectorAll('.xsel-opt');
      i = Math.max(0, Math.min(opts.length - 1, i)); open.active = i;
      opts.forEach((o, j) => o.classList.toggle('act', j === i));
      if (scroll && opts[i]) opts[i].scrollIntoView({block:'nearest'});
    }
    btn.addEventListener('click', show);
    btn.addEventListener('keydown', e => {
      const isOpen = open && open.sel === sel;
      if (['ArrowDown', 'ArrowUp'].includes(e.key)){
        e.preventDefault();
        if (!isOpen) return show();
        focusOpt(open.active + (e.key === 'ArrowDown' ? 1 : -1), true);
      } else if (e.key === 'Enter' || e.key === ' '){
        if (isOpen){ e.preventDefault(); pick(open.active); }
      } else if (e.key === 'Escape' || e.key === 'Tab'){
        close();
      } else if (isOpen && e.key.length === 1){
        const k = e.key.toLowerCase(), n = sel.options.length;
        for (let j = 1; j <= n; j++){ const idx = (open.active + j) % n; if (sel.options[idx].text.toLowerCase().startsWith(k)){ focusOpt(idx, true); break; } }
      }
    });
  }
  window.AMLUI = {enhanceSelects(root = document){ root.querySelectorAll('select').forEach(enhance); }};
})();
