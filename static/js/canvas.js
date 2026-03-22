/* ========================================
   canvas.js - Neural Network Canvas Renderer
   Draws neurons, weights (connections),
   forward/backward pass particle animations,
   and hover tooltips.
   ======================================== */

class CanvasRenderer {
    constructor(canvasId, mlp) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.mlp = mlp;
        this.dpr = window.devicePixelRatio || 1;

        // Layout
        this.neuronPositions = []; // [layer][index] = {x, y}
        this.neuronRadius = 28;

        // Animation particles
        this.particles = [];
        this.animating = false;
        this.animationPhase = null; // 'forward', 'backward', null

        // Hover
        this.hoveredNeuron = null; // {layer, index}
        this.mouseX = 0;
        this.mouseY = 0;

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
        const layerLabels = ['Giriş', 'Gizli', 'Çıkış'];

        for (let l = 0; l < numLayers; l++) {
            this.neuronPositions[l] = [];
            const x = marginX + (usableW / (numLayers - 1)) * l;
            const count = layers[l];
            const totalH = (count - 1) * 90;
            const startY = (usableH - totalH) / 2;

            for (let n = 0; n < count; n++) {
                this.neuronPositions[l][n] = {
                    x: x,
                    y: startY + n * 90,
                    label: layerLabels[l]
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
                if (dx * dx + dy * dy < (this.neuronRadius + 5) * (this.neuronRadius + 5)) {
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
            html += `y = ${output.toFixed(5)}<br>`;
        }

        const net = this.mlp.getNet(layer, index);
        if (net !== null) {
            html += `net = ${net.toFixed(5)}<br>`;
        }

        const delta = this.mlp.getDelta(layer, index);
        if (delta !== null) {
            html += `<span style="color:#ff9f43">δ = ${delta.toFixed(6)}</span>`;
        }

        tooltip.innerHTML = html;
        tooltip.classList.remove('hidden');

        // Position tooltip
        let tx = pos.x + this.neuronRadius + 15;
        let ty = pos.y - 30;
        if (tx + 180 > this.width) tx = pos.x - 195;
        if (ty < 10) ty = 10;
        tooltip.style.left = tx + 'px';
        tooltip.style.top = ty + 'px';
    }

    _hideTooltip() {
        const tooltip = document.getElementById('tooltip');
        if (tooltip) tooltip.classList.add('hidden');
    }

    // ---------- Drawing ----------

    draw() {
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

                    // Thickness proportional to |w|
                    const absW = Math.abs(w);
                    const thickness = Math.max(1, (absW / (maxW || 1)) * 6);

                    // Color: positive=blue, negative=red
                    const color = w >= 0 ? this.colors.weightPositive : this.colors.weightNegative;
                    const alpha = Math.min(0.9, 0.2 + (absW / (maxW || 1)) * 0.7);

                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.lineTo(to.x, to.y);
                    ctx.strokeStyle = color;
                    ctx.globalAlpha = alpha;
                    ctx.lineWidth = thickness;
                    ctx.stroke();
                    ctx.globalAlpha = 1.0;

                    // Weight value label at midpoint
                    const mx = (from.x + to.x) / 2;
                    const my = (from.y + to.y) / 2;
                    ctx.font = '10px "JetBrains Mono", monospace';
                    ctx.fillStyle = color;
                    ctx.globalAlpha = 0.75;
                    ctx.textAlign = 'center';
                    ctx.fillText(w.toFixed(2), mx, my - 5);
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

                // Choose color by layer type
                let strokeColor, glowColor;
                if (l === 0) {
                    strokeColor = this.colors.neuronInputStroke;
                    glowColor = this.colors.neuronInputGlow;
                } else if (l === state.layers.length - 1) {
                    strokeColor = this.colors.neuronOutputStroke;
                    glowColor = this.colors.neuronOutputGlow;
                } else {
                    strokeColor = this.colors.neuronStroke;
                    glowColor = this.colors.neuronGlow;
                }

                // Outer glow
                const glowRadius = isHovered ? 25 : 15;
                const gradient = ctx.createRadialGradient(pos.x, pos.y, r * 0.5, pos.x, pos.y, r + glowRadius);
                gradient.addColorStop(0, glowColor);
                gradient.addColorStop(1, 'transparent');
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, r + glowRadius, 0, Math.PI * 2);
                ctx.fillStyle = gradient;
                ctx.fill();

                // Neuron body
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
                ctx.fillStyle = this.colors.neuronFill;
                ctx.fill();
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = isHovered ? 3 : 2;
                ctx.stroke();

                // Value inside neuron
                const output = this.mlp.getOutput(l, n);
                if (output !== null) {
                    ctx.font = 'bold 11px "JetBrains Mono", monospace';
                    ctx.fillStyle = this.colors.text;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(output.toFixed(3), pos.x, pos.y);
                }

                // Neuron label below
                const labels = ['x', 'h', 'y'];
                ctx.font = '10px "Inter", sans-serif';
                ctx.fillStyle = this.colors.textMuted;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(`${labels[l]}${l > 0 ? (n + 1) : (n + 1)}`, pos.x, pos.y + r + 6);
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
            ctx.fillText(labels[l], pos.x, 25);
        }
    }

    // ---------- Particle Animations ----------

    spawnForwardParticles() {
        this.animationPhase = 'forward';
        this.particles = [];
        const state = this.mlp.getNetworkState();

        for (let l = 0; l < state.layers.length - 1; l++) {
            for (let j = 0; j < state.layers[l + 1]; j++) {
                for (let i = 0; i < state.layers[l]; i++) {
                    const from = this.neuronPositions[l][i];
                    const to = this.neuronPositions[l + 1][j];

                    // Stagger particles by layer
                    const delay = l * 40;
                    for (let p = 0; p < 3; p++) {
                        this.particles.push({
                            x: from.x, y: from.y,
                            targetX: to.x, targetY: to.y,
                            progress: -(delay + p * 12) / 100,
                            speed: 0.015 + Math.random() * 0.008,
                            color: this.colors.particleForward,
                            size: 3 + Math.random() * 2,
                            layer: l
                        });
                    }
                }
            }
        }
    }

    spawnBackwardParticles() {
        this.animationPhase = 'backward';
        this.particles = [];
        const state = this.mlp.getNetworkState();

        for (let l = state.layers.length - 2; l >= 0; l--) {
            for (let j = 0; j < state.layers[l + 1]; j++) {
                for (let i = 0; i < state.layers[l]; i++) {
                    const from = this.neuronPositions[l + 1][j]; // backward: from output side
                    const to = this.neuronPositions[l][i];

                    const delay = (state.layers.length - 2 - l) * 40;
                    for (let p = 0; p < 3; p++) {
                        this.particles.push({
                            x: from.x, y: from.y,
                            targetX: to.x, targetY: to.y,
                            progress: -(delay + p * 12) / 100,
                            speed: 0.015 + Math.random() * 0.008,
                            color: this.colors.particleBackward,
                            size: 3 + Math.random() * 2,
                            layer: l
                        });
                    }
                }
            }
        }
    }

    updateParticles() {
        let aliveCount = 0;

        for (const p of this.particles) {
            p.progress += p.speed;

            if (p.progress >= 0 && p.progress <= 1) {
                const t = p.progress;
                p.x = p.x + (p.targetX - (p.x - (p.targetX - p.x) * (t - p.speed) / (1 - (t - p.speed) + 0.001))) * 0;
                // Simple linear interpolation
                const startX = p.targetX - (p.targetX - p.x) / (t || 0.001) * t;
                p.x = this._particleStartX(p) + (p.targetX - this._particleStartX(p)) * t;
                p.y = this._particleStartY(p) + (p.targetY - this._particleStartY(p)) * t;
            }

            if (p.progress < 1.0) aliveCount++;
        }

        return aliveCount > 0;
    }

    _particleStartX(p) {
        // Calculate the start position based on target and current position
        // We stored targetX/targetY, so we need the original from position
        // Let's recalculate: the from position was set when spawning
        // We need to store it. Let me fix this with fromX/fromY.
        return p.fromX || p.x;
    }

    _particleStartY(p) {
        return p.fromY || p.y;
    }

    _drawParticles(ctx) {
        for (const p of this.particles) {
            if (p.progress < 0 || p.progress > 1) continue;

            const t = p.progress;
            const x = p.fromX + (p.targetX - p.fromX) * t;
            const y = p.fromY + (p.targetY - p.fromY) * t;

            // Glow
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, p.size * 4);
            gradient.addColorStop(0, p.color);
            gradient.addColorStop(1, 'transparent');
            ctx.beginPath();
            ctx.arc(x, y, p.size * 4, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            // Core
            ctx.beginPath();
            ctx.arc(x, y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
        }
    }

    // Fixed particle spawn with stored from positions
    spawnForwardParticles() {
        this.animationPhase = 'forward';
        this.particles = [];
        const state = this.mlp.getNetworkState();

        for (let l = 0; l < state.layers.length - 1; l++) {
            for (let j = 0; j < state.layers[l + 1]; j++) {
                for (let i = 0; i < state.layers[l]; i++) {
                    const from = this.neuronPositions[l][i];
                    const to = this.neuronPositions[l + 1][j];
                    const delay = l * 40;

                    for (let p = 0; p < 3; p++) {
                        this.particles.push({
                            fromX: from.x, fromY: from.y,
                            targetX: to.x, targetY: to.y,
                            progress: -(delay + p * 12) / 100,
                            speed: 0.018 + Math.random() * 0.008,
                            color: this.colors.particleForward,
                            size: 2.5 + Math.random() * 2
                        });
                    }
                }
            }
        }
    }

    spawnBackwardParticles() {
        this.animationPhase = 'backward';
        this.particles = [];
        const state = this.mlp.getNetworkState();

        for (let l = state.layers.length - 2; l >= 0; l--) {
            for (let j = 0; j < state.layers[l + 1]; j++) {
                for (let i = 0; i < state.layers[l]; i++) {
                    const from = this.neuronPositions[l + 1][j];
                    const to = this.neuronPositions[l][i];
                    const delay = (state.layers.length - 2 - l) * 40;

                    for (let p = 0; p < 3; p++) {
                        this.particles.push({
                            fromX: from.x, fromY: from.y,
                            targetX: to.x, targetY: to.y,
                            progress: -(delay + p * 12) / 100,
                            speed: 0.018 + Math.random() * 0.008,
                            color: this.colors.particleBackward,
                            size: 2.5 + Math.random() * 2
                        });
                    }
                }
            }
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
