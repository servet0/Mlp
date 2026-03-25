/* ========================================
   decision_boundary.js - XOR Decision Boundary Heatmap
   Draws a 2D color map showing how the MLP
   classifies the [0,1]×[0,1] input space.
   ======================================== */

class DecisionBoundary {
    constructor(canvasId, mlp) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.mlp = mlp;
        this.resolution = 60; // grid resolution (60×60)
        this.imageData = null;

        this._setup();
    }

    _setup() {
        const resize = () => {
            const rect = this.canvas.parentElement.getBoundingClientRect();
            const size = Math.min(rect.width, rect.height);
            this.canvas.width = size;
            this.canvas.height = size;
            this.canvas.style.width = size + 'px';
            this.canvas.style.height = size + 'px';
            this.size = size;
        };
        window.addEventListener('resize', resize);
        resize();
    }

    draw() {
        const ctx = this.ctx;
        const size = this.size;
        const res = this.resolution;
        const cellW = size / res;
        const cellH = size / res;
        const padding = 30; // padding for axes
        const plotSize = size - padding - 10;
        const plotCellW = plotSize / res;
        const plotCellH = plotSize / res;

        ctx.clearRect(0, 0, size, size);

        // Draw heatmap
        for (let iy = 0; iy < res; iy++) {
            for (let ix = 0; ix < res; ix++) {
                const x1 = ix / (res - 1);
                const x2 = 1 - iy / (res - 1); // flip Y so 0 is bottom

                const output = this.mlp.forward([x1, x2]);
                const val = Math.max(0, Math.min(1, output[0]));

                // Color: 0 = deep blue, 1 = bright orange/yellow
                const r = Math.floor(val * 255);
                const g = Math.floor(val * 140 + (1 - val) * 40);
                const b = Math.floor((1 - val) * 220 + val * 30);

                ctx.fillStyle = `rgb(${r},${g},${b})`;
                ctx.fillRect(padding + ix * plotCellW, 5 + iy * plotCellH, plotCellW + 1, plotCellH + 1);
            }
        }

        // Draw XOR data points
        const xorPoints = [
            { x1: 0, x2: 0, d: 0 },
            { x1: 0, x2: 1, d: 1 },
            { x1: 1, x2: 0, d: 1 },
            { x1: 1, x2: 1, d: 0 }
        ];

        for (const pt of xorPoints) {
            const px = padding + pt.x1 * plotSize;
            const py = 5 + (1 - pt.x2) * plotSize;

            // Outer ring
            ctx.beginPath();
            ctx.arc(px, py, 10, 0, Math.PI * 2);
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2.5;
            ctx.stroke();

            // Inner fill
            ctx.beginPath();
            ctx.arc(px, py, 7, 0, Math.PI * 2);
            ctx.fillStyle = pt.d === 1 ? '#22d3ae' : '#ff4e6a';
            ctx.fill();

            // Label
            ctx.font = 'bold 10px "JetBrains Mono", monospace';
            ctx.fillStyle = '#fff';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(pt.d, px, py);
        }

        // Axes labels
        ctx.font = '10px "Inter", sans-serif';
        ctx.fillStyle = '#9090b8';
        ctx.textAlign = 'center';

        // X axis
        ctx.fillText('0', padding, size - 2);
        ctx.fillText('x₁', padding + plotSize / 2, size - 2);
        ctx.fillText('1', padding + plotSize, size - 2);

        // Y axis
        ctx.save();
        ctx.translate(10, 5 + plotSize / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('x₂', 0, 0);
        ctx.restore();
        ctx.fillText('0', 18, 5 + plotSize);
        ctx.fillText('1', 18, 12);

        // Title
        ctx.font = '600 11px "Inter", sans-serif';
        ctx.fillStyle = '#9090b8';
        ctx.textAlign = 'right';
        ctx.fillText('Karar Sınırı', size - 5, size - 3);

        // Color legend bar
        const legendW = 12;
        const legendH = plotSize;
        const legendX = size - legendW - 2;
        const legendY = 5;
        const legendGrad = ctx.createLinearGradient(0, legendY, 0, legendY + legendH);
        legendGrad.addColorStop(0, 'rgb(255,140,30)');
        legendGrad.addColorStop(0.5, 'rgb(128,90,125)');
        legendGrad.addColorStop(1, 'rgb(0,40,220)');
        ctx.fillStyle = legendGrad;
        ctx.fillRect(legendX, legendY, legendW, legendH);
        ctx.strokeStyle = '#2a2a5a';
        ctx.lineWidth = 1;
        ctx.strokeRect(legendX, legendY, legendW, legendH);

        ctx.font = '9px "JetBrains Mono"';
        ctx.fillStyle = '#9090b8';
        ctx.textAlign = 'right';
        ctx.fillText('1', legendX - 3, legendY + 8);
        ctx.fillText('0', legendX - 3, legendY + legendH);
    }
}
