// Análise de batidas: andamento + rastreamento das batidas reais (programação dinâmica)
// + estimativa do primeiro tempo do compasso pelo peso dos graves.
// Função pura: recebe as amostras mono e a taxa de amostragem.

function analyzeAudio(data, sr) {
  const hop = Math.round(sr / 200);      // ~5 ms por quadro
  const FPS = sr / hop;                  // taxa real de quadros (evita deriva)
  const n = Math.floor(data.length / hop);
  if (n < FPS * 4) return null;          // menos de 4 s de música

  // Envelope de energia em banda cheia e em graves (~150 Hz, bumbo e baixo)
  const a = Math.exp(-2 * Math.PI * 150 / sr);
  let y1 = 0, y2 = 0;
  const envFull = new Float32Array(n), envLow = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    let sf = 0, sl = 0;
    const st = f * hop;
    for (let j = st; j < st + hop; j++) {
      const x = data[j];
      y1 = (1 - a) * x + a * y1;
      y2 = (1 - a) * y1 + a * y2;
      sf += x * x; sl += y2 * y2;
    }
    envFull[f] = sf / hop;
    envLow[f] = sl / hop;
  }
  // janela de 25 ms centrada (graves têm ciclos longos) e compressão logarítmica
  function smoothLog(e) {
    const out = new Float32Array(n);
    for (let f = 0; f < n; f++) {
      let s = 0, c = 0;
      for (let d = -2; d <= 2; d++) { const k = f + d; if (k >= 0 && k < n) { s += e[k]; c++; } }
      out[f] = Math.log(1 + 1000 * Math.sqrt(s / c));
    }
    return out;
  }
  const eF = smoothLog(envFull), eL = smoothLog(envLow);
  envFull.set(eF); envLow.set(eL);

  // Ataques: só os aumentos de energia
  function onsetOf(env) {
    const o = new Float32Array(n);
    for (let f = 1; f < n; f++) o[f] = Math.max(0, env[f] - env[f - 1]);
    // remove a média local (1 s) e normaliza
    const W = Math.round(FPS), out = new Float32Array(n);
    let acc = 0;
    for (let f = 0; f < n; f++) {
      acc += o[f];
      if (f >= W) acc -= o[f - W];
      out[f] = o[f] - acc / Math.min(f + 1, W);
    }
    let s2 = 0;
    for (let f = 0; f < n; f++) s2 += out[f] * out[f];
    const sd = Math.sqrt(s2 / n) || 1;
    for (let f = 0; f < n; f++) out[f] /= sd;
    return out;
  }
  const onFull = onsetOf(envFull), onLow = onsetOf(envLow);
  const on = new Float32Array(n);
  for (let f = 0; f < n; f++) on[f] = onFull[f] + onLow[f];

  // Andamento: autocorrelação de 50 a 200 BPM, preferindo andamentos perto de 100
  const minLag = Math.floor(60 * FPS / 200), maxLag = Math.ceil(60 * FPS / 50);
  const ac = new Float64Array(maxLag + 2);
  for (let lag = minLag - 1; lag <= maxLag + 1; lag++) {
    let s = 0;
    for (let i = lag; i < n; i++) s += on[i] * on[i - lag];
    ac[lag] = s / (n - lag);
  }
  let bestLag = minLag, bestVal = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    const bpm = 60 * FPS / lag;
    const w = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 100) / 0.9, 2));
    // reforço com o dobro do período (o compasso se repete)
    const v = (ac[lag] + 0.5 * (ac[2 * lag] || 0)) * w;
    if (v > bestVal) { bestVal = v; bestLag = lag; }
  }
  // interpolação parabólica para precisão abaixo de um quadro
  const y0 = ac[bestLag - 1], yc = ac[bestLag], yp = ac[bestLag + 1];
  const den = y0 - 2 * yc + yp;
  const period = bestLag + (den !== 0 ? Math.max(-0.5, Math.min(0.5, 0.5 * (y0 - yp) / den)) : 0);

  // Rastreamento das batidas por programação dinâmica:
  // cada batida cai num ataque forte, e o espaço entre batidas fica perto do período
  const tight = 300;
  const lo = Math.max(1, Math.round(period / 2)), hi = Math.round(period * 2);
  const score = new Float64Array(n), back = new Int32Array(n).fill(-1);
  const pen = new Float64Array(hi + 1);
  for (let d = lo; d <= hi; d++) pen[d] = tight * Math.pow(Math.log(d / period), 2);
  for (let t = 0; t < n; t++) {
    let best = -Infinity, bi = -1;
    const start = Math.max(0, t - hi), end = t - lo;
    for (let p = start; p <= end; p++) {
      const v = score[p] - pen[t - p];
      if (v > best) { best = v; bi = p; }
    }
    score[t] = on[t] + (bi >= 0 && best > 0 ? best : 0);
    back[t] = bi >= 0 && best > 0 ? bi : -1;
  }
  let endT = n - 1, endBest = -Infinity;
  for (let t = Math.max(0, n - Math.round(period)); t < n; t++) {
    if (score[t] > endBest) { endBest = score[t]; endT = t; }
  }
  const frames = [];
  for (let t = endT; t >= 0; t = back[t]) frames.push(t);
  frames.reverse();
  // ajuste fino: cada batida vai para o pico de ataque mais próximo (±30 ms)
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    let bf = f, bv = on[f];
    for (let d = -6; d <= 6; d++) {
      const k = f + d;
      if (k > 0 && k < n && on[k] > bv) { bv = on[k]; bf = k; }
    }
    frames[i] = bf;
  }
  // descarta batidas fracas no começo e no fim (introdução ou silêncio)
  const strength = f => { let m = -Infinity; for (let d = -6; d <= 6; d++) { const v = on[f + d]; if (v > m) m = v; } return m; };
  while (frames.length && strength(frames[0]) < 2) frames.shift();
  while (frames.length && strength(frames[frames.length - 1]) < 2) frames.pop();
  if (frames.length < 8) return null;
  // o ataque começa um pouco antes do pico da diferença de energia
  const beats = frames.map(f => (f - 0.5) / FPS);

  // Andamento final: média ao longo da música inteira
  const bpm = 60 * (beats.length - 1) / (beats[beats.length - 1] - beats[0]);

  // Primeiro tempo do compasso: a posição (de 4) com graves mais fortes
  const phaseScore = [0, 0, 0, 0];
  frames.forEach((f, i) => {
    let m = 0;
    for (let d = -3; d <= 3; d++) { const v = onLow[f + d]; if (v > m) m = v; }
    phaseScore[i % 4] += m;
  });
  let downbeat = 0;
  for (let k = 1; k < 4; k++) if (phaseScore[k] > phaseScore[downbeat]) downbeat = k;

  return { beats, bpm, downbeat };
}

if (typeof module !== "undefined") module.exports = { analyzeAudio };
