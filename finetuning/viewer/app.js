(() => {
  const COLORS = { diffsbdd: "cyanCarbon", grpo: "orangeCarbon" };
  const HALO = { diffsbdd: "#2fb6c8", grpo: "#f08b34" };
  const PROTEIN_MODEL = 0;
  const LIGAND_MODEL = 1;
  const SITE_MODEL = 2;
  const state = {
    pocket: 0,
    model: "grpo",
    ligand: 0,
    rep: "surface",
    manifest: null,
    viewer: null,
    loadedKey: "",
    pocketChartKey: "",
    ligandChartKey: "",
    request: 0,
  };

  const $ = (id) => document.getElementById(id);

  function elementColor(atom) {
    const colors = {
      C: 0x8ccfd0,
      O: 0xef6c67,
      N: 0x666bd1,
      S: 0xe5d45a,
      P: 0xe6a65d,
      H: 0xf0f0f0,
    };
    return colors[atom.elem] || 0xaab4b8;
  }

  function viewer() {
    if (state.viewer) return state.viewer;
    state.viewer = $3Dmol.createViewer($("stage"), {
      backgroundColor: "0xffffff",
      antialias: true,
      cartoonQuality: 10,
    });
    bindWheelZoom(state.viewer);
    $("stage").addEventListener("dblclick", () => {
      if (!hasLigand()) return;
      state.viewer.zoomTo({ model: LIGAND_MODEL }, 400);
      state.viewer.zoom(0.5, 400);
    });
    return state.viewer;
  }

  function bindWheelZoom(v) {
    document.querySelector("main").addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        event.stopPropagation();
        v.zoom(event.deltaY < 0 ? 1.12 : 1 / 1.12, 80);
      },
      { capture: true, passive: false },
    );
  }

  const textCache = new Map();

  async function loadText(url) {
    if (textCache.has(url)) return textCache.get(url);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not load ${url}`);
    const text = await res.text();
    textCache.set(url, text);
    return text;
  }

  function ligandsFor() {
    return state.manifest.pockets[state.pocket].models[state.model] || [];
  }


  function wantsFullProtein() {
    return state.rep === "ribbon" && Boolean(state.manifest.pockets[state.pocket].pdb_full);
  }

  function sceneKey() {
    return `${state.pocket}|${state.model}|${state.ligand}|${wantsFullProtein()}`;
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, Number(value) || 0));
  }

  function addComparison(parent, label, left, right, normalize, format) {
    const row = document.createElement("div");
    row.className = "mini-chart";
    const heading = document.createElement("span");
    heading.className = "chart-title";
    heading.textContent = label;
    const bars = document.createElement("div");
    bars.className = "vertical-bars";
    [
      ["diff", left],
      ["grpo", right],
    ].forEach(([kind, value]) => {
      const column = document.createElement("div");
      column.className = "bar-column";
      const number = document.createElement("span");
      number.className = "vertical-value";
      number.textContent = value == null ? "—" : format(value);
      const leader = document.createElement("span");
      leader.className = "leader";
      const pillar = document.createElement("div");
      pillar.className = `pillar ${kind}`;
      column.style.setProperty("--bar-height", `${Math.max(5, 70 * clamp01(normalize(value)))}%`);
      const shortLabel = document.createElement("span");
      shortLabel.className = "bar-label";
      shortLabel.textContent = kind === "diff" ? "DiffSBDD" : "GRPO";
      column.append(number, leader, pillar, shortLabel);
      bars.appendChild(column);
    });
    row.append(heading, bars);
    parent.appendChild(row);
  }

  const vinaScale = (v) => (5 - v) / 13;
  const twoDecimals = (v) => v.toFixed(2);


  function updatePocketCharts(pocket) {
    if (state.pocketChartKey === String(state.pocket)) return;
    state.pocketChartKey = String(state.pocket);

    const chart = $("pocket-chart");
    chart.innerHTML = "";
    const diff = pocket.heldout?.diffsbdd;
    const grpo = pocket.heldout?.grpo;
    if (!diff || !grpo) {
      chart.textContent = "Pocket summary unavailable.";
      return;
    }
    addComparison(chart, "Vina", diff.vina, grpo.vina, vinaScale, twoDecimals);
    addComparison(chart, "QED", diff.qed, grpo.qed, (v) => v, twoDecimals);
    addComparison(chart, "SA", diff.sa, grpo.sa, (v) => v, twoDecimals);
    addComparison(
      chart,
      "Validity",
      diff.validity,
      grpo.validity,
      (v) => v,
      (v) => `${Math.round(v * 100)}%`,
    );
  }

  function updateLigandCharts(pocket) {
    const key = `${state.pocket}|${state.ligand}`;
    if (state.ligandChartKey === key) return;
    state.ligandChartKey = key;

    $("ligand-heading").textContent = `Ligand ${state.ligand + 1}`;
    const chart = $("ligand-chart");
    chart.innerHTML = "";
    const diff = pocket.models.diffsbdd?.[state.ligand];
    const grpo = pocket.models.grpo?.[state.ligand];
    const pick = (item, field) => (item ? item[field] : null);

    addComparison(chart, "Vina", pick(diff, "vina"), pick(grpo, "vina"), vinaScale, twoDecimals);
    addComparison(chart, "QED", pick(diff, "qed"), pick(grpo, "qed"), (v) => v, twoDecimals);
    addComparison(chart, "SA", pick(diff, "sa"), pick(grpo, "sa"), (v) => v, twoDecimals);
    addComparison(
      chart,
      "Atoms",
      pick(diff, "atom_count"),
      pick(grpo, "atom_count"),
      (v) => v / 35,
      (v) => String(Math.round(v)),
    );
  }

  function syncHud() {
    const pocket = state.manifest.pockets[state.pocket];
    const ligs = ligandsFor();
    if (state.ligand >= ligs.length) state.ligand = 0;
    const source = wantsFullProtein() ? `full ${pocket.full_pdb_id}` : "pocket";
    const label = state.model === "grpo" ? "GRPO fine-tuned" : "DiffSBDD";
    $("status").textContent = `${pocket.id} · ${label} · ${source}`;
    $("btn-diffsbdd").classList.toggle("active", state.model === "diffsbdd");
    $("btn-grpo").classList.toggle("active", state.model === "grpo");
    document.querySelectorAll("#rep [data-rep]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.rep === state.rep);
    });
    const box = $("ligands");
    box.innerHTML = "";
    ligs.forEach((_, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = String(i + 1);
      btn.classList.toggle("active", i === state.ligand);
      btn.addEventListener("click", () => { state.ligand = i; draw(false); });
      box.appendChild(btn);
    });
    $("pocket-label").textContent = pocket.id;
    $("pocket-menu").querySelectorAll("li").forEach((item, i) => {
      item.setAttribute("aria-selected", String(i === state.pocket));
    });
    updatePocketCharts(pocket);
    updateLigandCharts(pocket);
  }

  function styleLigand(v, bold) {
    v.setStyle({ model: LIGAND_MODEL }, {
      stick: { radius: bold ? 0.32 : 0.23, colorscheme: COLORS[state.model] },
      sphere: { scale: bold ? 0.45 : 0.34, colorscheme: COLORS[state.model] },
    });
  }

  function siteResidues(v, cutoff) {
    const ligand = v.getModel(LIGAND_MODEL).selectedAtoms({});
    const reach = cutoff * cutoff;
    const byChain = new Map();
    v.getModel(PROTEIN_MODEL).selectedAtoms({}).forEach((atom) => {
      const touches = ligand.some((other) => {
        const dx = other.x - atom.x;
        const dy = other.y - atom.y;
        const dz = other.z - atom.z;
        return dx * dx + dy * dy + dz * dz <= reach;
      });
      if (!touches) return;
      const chain = atom.chain || "";
      if (!byChain.has(chain)) byChain.set(chain, new Set());
      byChain.get(chain).add(atom.resi);
    });
    return byChain;
  }

  function applyRibbon(v, hasLig) {
    v.setStyle({ model: PROTEIN_MODEL }, {
      cartoon: {
        color: "spectrum",
        style: "oval",
        arrows: true,
        tubes: true,
        thickness: 0.8,
        opacity: 0.45,
      },
    });
    if (!hasLig) return;
    siteResidues(v, 6).forEach((residues, chain) => {
      const sel = { model: SITE_MODEL, resi: Array.from(residues) };
      if (chain) sel.chain = chain;
      v.setStyle(sel, { stick: { radius: 0.13, color: "#5d6f7e" } });
    });
    v.addSurface(
      $3Dmol.SurfaceType.SAS,
      { opacity: 0.14, color: HALO[state.model] },
      { model: LIGAND_MODEL },
    );
    v.addSurface(
      $3Dmol.SurfaceType.VDW,
      { opacity: 0.3, color: HALO[state.model] },
      { model: LIGAND_MODEL },
    );
  }

  function applyRepresentation(v, hasLig) {
    v.setStyle({ model: PROTEIN_MODEL }, {});
    if (hasLig) v.setStyle({ model: SITE_MODEL }, {});
    v.removeAllSurfaces();
    if (state.rep === "ribbon") {
      applyRibbon(v, hasLig);
    } else {
      v.addSurface(
        $3Dmol.SurfaceType.MS,
        { opacity: 0.82, colorfunc: elementColor },
        { model: PROTEIN_MODEL },
      );
    }
    if (hasLig) styleLigand(v, state.rep === "ribbon");
  }

  async function ensureModels(v) {
    const key = sceneKey();
    if (state.loadedKey === key) return;
    v.removeAllModels();
    v.removeAllSurfaces();
    const pocket = state.manifest.pockets[state.pocket];
    const protein = await loadText(wantsFullProtein() ? pocket.pdb_full : pocket.pdb);
    v.addModel(protein, "pdb");
    const ligs = ligandsFor();
    if (ligs.length) {
      v.addModel(await loadText(ligs[state.ligand].sdf), "sdf");

      v.addModel(protein, "pdb");
    }
    state.loadedKey = key;
  }

  function hasLigand() {
    return ligandsFor().length > 0;
  }

  async function draw(resetView) {
    if (!state.manifest) return;
    const request = ++state.request;
    syncHud();
    const v = viewer();
    await ensureModels(v);
    if (request !== state.request) return;
    applyRepresentation(v, hasLigand());
    v.resize();
    if (resetView) v.zoomTo();
    v.render();
    setTimeout(() => v.render(), 80);
  }

  function buildDropdown() {
    const dropdown = $("pocket-dropdown");
    const menu = $("pocket-menu");
    const close = () => dropdown.classList.remove("open");

    state.manifest.pockets.forEach((pocket, i) => {
      const item = document.createElement("li");
      item.role = "option";
      item.textContent = pocket.id;
      item.addEventListener("click", () => {
        close();
        if (i === state.pocket) return;
        state.pocket = i;
        state.ligand = 0;
        draw(true);
      });
      menu.appendChild(item);
    });

    $("pocket-toggle").addEventListener("click", (event) => {
      event.stopPropagation();
      const open = dropdown.classList.toggle("open");
      $("pocket-toggle").setAttribute("aria-expanded", String(open));
    });
    document.addEventListener("click", close);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") close();
    });
  }

  async function boot() {
    try {
      state.manifest = await (await fetch("data/manifest.json")).json();
    } catch (err) {
      $("status").textContent = "Missing data/manifest.json";
      return;
    }
    buildDropdown();
    if (!state.manifest.pockets[0]?.models.grpo?.length) state.model = "diffsbdd";

    $("btn-diffsbdd").addEventListener("click", () => { state.model = "diffsbdd"; draw(true); });
    $("btn-grpo").addEventListener("click", () => { state.model = "grpo"; draw(true); });
    document.querySelectorAll("#rep [data-rep]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const wasFull = state.rep === "ribbon";
        state.rep = btn.dataset.rep;
        draw(wasFull !== (state.rep === "ribbon"));
      });
    });
    window.addEventListener("resize", () => { viewer().resize(); viewer().render(); });
    await draw(true);
  }

  boot().catch((err) => { $("status").textContent = String(err); console.error(err); });
})();
