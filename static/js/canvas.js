/* ========================================
   canvas.js - Neural Network Canvas Renderer
   Draws neurons with activation heat coloring,
   weight connections, gradient flow visualization,
   forward/backward pass particle animations,
   bias labels, and hover tooltips.
   ======================================== */

class CanvasRenderer {
    constructor(canvasId, mlp) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.mlp = mlp;
        this.dpr = window.devicePixelRatio || 1;

        // Layout
        this.neuronPositions = []; // [layer][index] = {x, y}
        this.neuronRadius = 30;

        // Animation particles
        this.particles = [];
        this.animationPhase = null; // 'forward', 'backward', null

        // Gradient flow mode
        this.showGradientFlow = false;

        // Hover
        this.hoveredNeuron = null;
        this.mouseX = 0;
        this.mouseY = 0;

        // Time for ambient animation
        this.time = 0;

        // Colors
        this.colors = {
            neuronFill: '#141432',
            neuronStroke: '#00e5ff',
            neuronGlow: 'rgba(0, 229, 255, 0.4)',
            neuronInputStroke: '#a855f7',
            neuronInputGlow: 'rgba(168, 85, 247, 0.4)',
            neuronOutputStroke: '#22d3ae',
            neuronOutputGlow: 'rgba(34, 211, 174, 0.4)',
            weightPositive: '#4ea8ff',
            weightNegative: '#ff4e6a',
            particleForward: '#00e5ff',
            particleBackward: '#ff9f43',
            text: '#e8e8f0',
            textMuted: '#9090b8',
            bg: '#0f1029'
        };

        this._setupCanvas();
        this._setupMouseTracking();
    }

    _setupCanvas() {
        const resize = () => {
            const rect = this.canvas.parentElement.getBoundingClientRect();
            this.canvas.width = rect.width * this.dpr;
            this.canvas.height = rect.height * this.dpr;
            this.canvas.style.width = rect.width + 'px';
            this.canvas.style.height = rect.height + 'px';
            this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
            this.width = rect.width;
            this.height = rect.height;
            this._computeLayout();
        };
        window.addEventListener('resize', resize);
        resize();
    }

    _setupMouseTracking() {
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouseX = e.clientX - rect.left;
            this.mouseY = e.clientY - rect.top;
            this._checkHover();
        });
        this.canvas.addEventListener('mouseleave', () => {
            this.hoveredNeuron = null;
            this._hideTooltip();
        });
    }

    _computeLayout() {
        const layers = this.mlp.layers;
        const numLayers = layers.length;
        const marginX = 120;
        const usableW = this.width - marginX * 2;
        const usableH = this.height;

        this.neuronPositions = [];

        for (let l = 0; l < numLayers; l++) {
            this.neuronPositions[l] = [];
            const x = marginX + (usableW / (numLayers - 1)) * l;
            const count = layers[l];
            const totalH = (count - 1) * 100;
            const startY = (usableH - totalH) / 2;

            for (let n = 0; n < count; n++) {
                this.neuronPositions[l][n] = {
                    x: x,
                    y: startY + n * 100
                };
            }
        }
    }

    _checkHover() {
        this.hoveredNeuron = null;
        for (let l = 0; l < this.neuronPositions.length; l++) {
            for (let n = 0; n < this.neuronPositions[l].length; n++) {
                const pos = this.neuronPositions[l][n];
                const dx = this.mouseX - pos.x;
                const dy = this.mouseY - pos.y;
                if (dx * dx + dy * dy < (this.neuronRadius + 8) * (this.neuronRadius + 8)) {
                    this.hoveredNeuron = { layer: l, index: n };
                    this._showTooltip(l, n, pos);
                    return;
                }
            }
        }
        this._hideTooltip();
    }

    _showTooltip(layer, index, pos) {
        const tooltip = document.getElementById('tooltip');
        if (!tooltip) return;

        let html = '';
        const layerNames = ['Giriş', 'Gizli', 'Çıkış'];
        html += `<strong>${layerNames[layer]} Katmanı [${index}]</strong><br>`;

        const output = this.mlp.getOutput(layer, index);
        if (output !== null) {
            html += `<span style="color:#00e5ff">y = ${output.toFixed(5)}</span><br>`;
        }

        const net = this.mlp.getNet(layer, index);
        if (net !== null) {
            html += `net = ${net.toFixed(5)}<br>`;
        }

        // Show bias for non-input neurons
        if (layer > 0) {
            const bias = this.mlp.biases[layer - 1]?.[index];
            if (bias !== undefined) {
                html += `<span style="color:#a855f7">bias = ${bias.toFixed(5)}</span><br>`;
            }
        }

        const delta = this.mlp.getDelta(layer, index);
        if (delta !== null) {
            html += `<span style="color:#ff9f43">δ = ${delta.toFixed(6)}</span><br>`;

            // Show activation derivative
            if (output !== null) {
                const phiPrime = this.mlp.activationDerivative(output);
                html += `<span style="color:#606088">φ'(v) = ${phiPrime.toFixed(6)}</span>`;
            }
        }

        tooltip.innerHTML = html;
        tooltip.classList.remove('hidden');

        let tx = pos.x + this.neuronRadius + 18;
        let ty = pos.y - 40;
        if (tx + 200 > this.width) tx = pos.x - 215;
        if (ty < 10) ty = 10;
        tooltip.style.left = tx + 'px';
        tooltip.style.top = ty + 'px';
    }

    _hideTooltip() {
        const tooltip = document.getElementById('tooltip');
        if (tooltip) tooltip.classList.add('hidden');
    }

    // --- Activation Heat Color ---
    _activationColor(value) {
        // Maps activation [0,1] to cold→hot gradient
        // 0.0 = deep blue, 0.5 = purple/neutral, 1.0 = bright orange
        const t = Math.max(0, Math.min(1, value));
        let r, g, b;
        if (t < 0.5) {
            const s = t / 0.5;
            r = Math.floor(30 + s * 100);
            g = Math.floor(40 + s * 30);
            b = Math.floor(200 - s * 80);
        } else {
            const s = (t - 0.5) / 0.5;
            r = Math.floor(130 + s * 125);
            g = Math.floor(70 + s * 100);
            b = Math.floor(120 - s * 90);
        }
        return `rgb(${r},${g},${b})`;
    }

    _activationGlow(value) {
        const t = Math.max(0, Math.min(1, value));
        if (t < 0.5) {
            return `rgba(30,40,200,${0.2 + t * 0.4})`;
        }
        return `rgba(255,170,30,${0.2 + (t - 0.5) * 0.8})`;
    }

    // ---------- Drawing ----------

    draw() {
        this.time += 0.02;
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);

        this._drawConnections(ctx);
        this._drawParticles(ctx);
        this._drawNeurons(ctx);
        this._drawLayerLabels(ctx);
    }

    _drawConnections(ctx) {
        const state = this.mlp.getNetworkState();

        for (let l = 0; l < state.layers.length - 1; l++) {
            const maxW = this._getMaxWeight(state.weights[l]);

            for (let j = 0; j < state.layers[l + 1]; j++) {
                for (let i = 0; i < state.layers[l]; i++) {
                    const w = state.weights[l][j][i];
                    const from = this.neuronPositions[l][i];
                    const to = this.neuronPositions[l + 1][j];

                    const absW = Math.abs(w);
                    const thickness = Math.max(1, (absW / (maxW || 1)) * 7);
                    const color = w >= 0 ? this.colors.weightPositive : this.colors.weightNegative;
                    const alpha = Math.min(0.9, 0.15 + (absW / (maxW || 1)) * 0.75);

                    // Gradient flow glow (during/after backward pass)
                    if (this.showGradientFlow && state.deltas.length > 0) {
                        const deltaIdx = l;  // delta index for layer l+1
                        const delta = state.deltas[deltaIdx]?.[j];
                        if (delta !== undefined) {
                            const absDelta = Math.abs(delta);
                            const glowIntensity = Math.min(1, absDelta * 8);
                            if (glowIntensity > 0.05) {
                                ctx.save();
                                ctx.beginPath();
                                ctx.moveTo(from.x, from.y);
                                ctx.lineTo(to.x, to.y);
                                ctx.strokeStyle = '#ff9f43';
                                ctx.globalAlpha = glowIntensity * 0.5;
                                ctx.lineWidth = thickness + 6;
                                ctx.shadowColor = '#ff9f43';
                                ctx.shadowBlur = 15;
                                ctx.stroke();
                                ctx.restore();
                            }
                        }
                    }

                    // Main connection line
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.lineTo(to.x, to.y);
                    ctx.strokeStyle = color;
                    ctx.globalAlpha = alpha;
                    ctx.lineWidth = thickness;
                    ctx.stroke();
                    ctx.globalAlpha = 1.0;

                    // Weight label at midpoint with offset to avoid overlap
                    const offsetY = (i - (state.layers[l] - 1) / 2) * 12;
                    const mx = (from.x + to.x) / 2;
                    const my = (from.y + to.y) / 2 + offsetY;
                    ctx.font = '10px "JetBrains Mono", monospace';
                    ctx.fillStyle = color;
                    ctx.globalAlpha = 0.8;
                    ctx.textAlign = 'center';
                    ctx.fillText(w.toFixed(2), mx, my - 6);
                    ctx.globalAlpha = 1.0;
                }
            }
        }
    }

    _getMaxWeight(layerWeights) {
        let max = 0;
        for (const row of layerWeights) {
            for (const w of row) {
                const absW = Math.abs(w);
                if (absW > max) max = absW;
            }
        }
        return max || 1;
    }

    _drawNeurons(ctx) {
        const state = this.mlp.getNetworkState();
        const r = this.neuronRadius;

        for (let l = 0; l < state.layers.length; l++) {
            for (let n = 0; n < state.layers[l]; n++) {
                const pos = this.neuronPositions[l][n];
                const isHovered = this.hoveredNeuron &&
                    this.hoveredNeuron.layer === l &&
                    this.hoveredNeuron.index === n;

                const output = this.mlp.getOutput(l, n);

                // Choose colors
                let strokeColor, glowColor, fillColor;
                if (l === 0) {
                    strokeColor = this.colors.neuronInputStroke;
                    glowColor = this.colors.neuronInputGlow;
                    fillColor = this.colors.neuronFill;
                } else if (l === state.layers.length - 1) {
                    strokeColor = this.colors.neuronOutputStroke;
                    glowColor = this.colors.neuronOutputGlow;
                    // Heat color for output
                    fillColor = output !== null ? this._activationColor(output) : this.colors.neuronFill;
                } else {
                    strokeColor = this.colors.neuronStroke;
                    glowColor = this.colors.neuronGlow;
                    // Heat color for hidden
                    fillColor = output !== null ? this._activationColor(output) : this.colors.neuronFill;
                }

                // Activation-based glow
                const activGlow = (output !== null && l > 0) ? this._activationGlow(output) : glowColor;
                const glowRadius = isHovered ? 28 : 18;
                const pulseOffset = Math.sin(this.time * 2 + l + n) * 3;

                const gradient = ctx.createRadialGradient(
                    pos.x, pos.y, r * 0.3,
                    pos.x, pos.y, r + glowRadius + pulseOffset
                );
                gradient.addColorStop(0, activGlow);
                gradient.addColorStop(1, 'transparent');
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, r + glowRadius + pulseOffset, 0, Math.PI * 2);
                ctx.fillStyle = gradient;
                ctx.fill();

                // Neuron body — activation heat fill
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
                ctx.fillStyle = fillColor;
                ctx.fill();
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = isHovered ? 3.5 : 2;
                ctx.stroke();

                // Inner ring (subtle)
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, r - 4, 0, Math.PI * 2);
                ctx.strokeStyle = strokeColor;
                ctx.globalAlpha = 0.15;
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.globalAlpha = 1.0;

                // Value inside neuron
                if (output !== null) {
                    ctx.font = 'bold 12px "JetBrains Mono", monospace';
                    ctx.fillStyle = '#fff';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(output.toFixed(3), pos.x, pos.y);
                }

                // Neuron label & bias below
                const labels = ['x', 'h', 'y'];
                const neuronLabel = `${labels[l]}${n + 1}`;
                ctx.font = '11px "Inter", sans-serif';
                ctx.fillStyle = this.colors.textMuted;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(neuronLabel, pos.x, pos.y + r + 6);

                // Show bias for hidden/output neurons
                if (l > 0) {
                    const bias = this.mlp.biases[l - 1]?.[n];
                    if (bias !== undefined) {
                        ctx.font = '9px "JetBrains Mono", monospace';
                        ctx.fillStyle = '#a855f7';
                        ctx.globalAlpha = 0.7;
                        ctx.fillText(`b=${bias.toFixed(2)}`, pos.x, pos.y + r + 20);
                        ctx.globalAlpha = 1.0;
                    }
                }

                // Delta badge (show during/after backward)
                if (l > 0) {
                    const delta = this.mlp.getDelta(l, n);
                    if (delta !== null && this.showGradientFlow) {
                        const absDelta = Math.abs(delta);
                        const badgeAlpha = Math.min(1, 0.4 + absDelta * 6);

                        ctx.save();
                        ctx.globalAlpha = badgeAlpha;
                        ctx.font = 'bold 9px "JetBrains Mono", monospace';
                        ctx.fillStyle = '#ff9f43';
                        ctx.textAlign = 'center';
                        ctx.fillText(`δ=${delta.toFixed(4)}`, pos.x, pos.y - r - 10);
                        ctx.restore();
                    }
                }
            }
        }
    }

    _drawLayerLabels(ctx) {
        const layers = this.mlp.layers;
        const labels = ['Giriş Katmanı', 'Gizli Katman', 'Çıkış Katmanı'];

        for (let l = 0; l < layers.length; l++) {
            const pos = this.neuronPositions[l][0];
            ctx.font = '600 12px "Inter", sans-serif';
            ctx.fillStyle = this.colors.textMuted;
            ctx.textAlign = 'center';
            ctx.fillText(labels[l], pos.x, 20);
        }

        // Show activation function label
        const fnLabel = this.mlp.activationFn === 'tanh' ? 'φ = tanh' : 'φ = sigmoid';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillStyle = '#606088';
        ctx.textAlign = 'left';
        ctx.fillText(fnLabel, 10, this.height - 10);
    }

    // ---------- Particle Animations ----------

    spawnForwardParticles() {
        this.animationPhase = 'forward';
        this.showGradientFlow = false;
        this.particles = [];
        const state = this.mlp.getNetworkState();

        for (let l = 0; l < state.layers.length - 1; l++) {
            for (let j = 0; j < state.layers[l + 1]; j++) {
                for (let i = 0; i < state.layers[l]; i++) {
                    const from = this.neuronPositions[l][i];
                    const to = this.neuronPositions[l + 1][j];
                    const delay = l * 45;

                    for (let p = 0; p < 4; p++) {
                        this.particles.push({
                            fromX: from.x, fromY: from.y,
                            targetX: to.x, targetY: to.y,
                            progress: -(delay + p * 10) / 100,
                            speed: 0.016 + Math.random() * 0.006,
                            color: this.colors.particleForward,
                            size: 2 + Math.random() * 2.5,
                            trail: []
                        });
                    }
                }
            }
        }
    }

    spawnBackwardParticles() {
        this.animationPhase = 'backward';
        this.showGradientFlow = true;
        this.particles = [];
        const state = this.mlp.getNetworkState();

        for (let l = state.layers.length - 2; l >= 0; l--) {
            for (let j = 0; j < state.layers[l + 1]; j++) {
                for (let i = 0; i < state.layers[l]; i++) {
                    const from = this.neuronPositions[l + 1][j];
                    const to = this.neuronPositions[l][i];
                    const delay = (state.layers.length - 2 - l) * 45;

                    // Particle size proportional to delta magnitude
                    const deltaIdx = l;
                    const delta = state.deltas[deltaIdx]?.[j];
                    const absDelta = delta ? Math.abs(delta) : 0.1;
                    const particleSize = 2 + Math.min(4, absDelta * 20);

                    for (let p = 0; p < 4; p++) {
                        this.particles.push({
                            fromX: from.x, fromY: from.y,
                            targetX: to.x, targetY: to.y,
                            progress: -(delay + p * 10) / 100,
                            speed: 0.016 + Math.random() * 0.006,
                            color: this.colors.particleBackward,
                            size: particleSize + Math.random() * 1.5,
                            trail: []
                        });
                    }
                }
            }
        }
    }

    _drawParticles(ctx) {
        for (const p of this.particles) {
            if (p.progress < 0 || p.progress > 1) continue;

            const t = p.progress;
            const x = p.fromX + (p.targetX - p.fromX) * t;
            const y = p.fromY + (p.targetY - p.fromY) * t;

            // Store trail positions
            p.trail.push({ x, y, alpha: 1 });
            if (p.trail.length > 8) p.trail.shift();

            // Draw trail
            for (let i = 0; i < p.trail.length; i++) {
                const tp = p.trail[i];
                const ta = (i / p.trail.length) * 0.4;
                ctx.beginPath();
                ctx.arc(tp.x, tp.y, p.size * (0.3 + i / p.trail.length * 0.7), 0, Math.PI * 2);
                ctx.fillStyle = p.color;
                ctx.globalAlpha = ta;
                ctx.fill();
            }
            ctx.globalAlpha = 1.0;

            // Main glow
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, p.size * 5);
            gradient.addColorStop(0, p.color);
            gradient.addColorStop(1, 'transparent');
            ctx.beginPath();
            ctx.arc(x, y, p.size * 5, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            // Core dot
            ctx.beginPath();
            ctx.arc(x, y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
        }
    }

    updateParticles() {
        let alive = 0;
        for (const p of this.particles) {
            p.progress += p.speed;
            if (p.progress < 1.0) alive++;
        }
        return alive > 0;
    }

    hasParticles() {
        return this.particles.some(p => p.progress < 1.0);
    }

    clearParticles() {
        this.particles = [];
        this.animationPhase = null;
    }
}
