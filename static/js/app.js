/* ========================================
   app.js - Main Application Controller
   Ties together MLP engine, Canvas renderer,
   Decision Boundary, Math Board, Chart.js,
   and all UI controls.
   ======================================== */

document.addEventListener('DOMContentLoaded', () => {
    // --- XOR training data ---
    const XOR_DATA = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ];

    let trainingData = [...XOR_DATA];
    let epoch = 0;
    let isTraining = false;
    let trainInterval = null;
    let animationFrameId = null;
    let epochsPerTick = 10;
    let hasConverged = false;

    // --- Initialize MLP, Renderers ---
    const mlp = new MLP();
    const renderer = new CanvasRenderer('network-canvas', mlp);
    const boundary = new DecisionBoundary('boundary-canvas', mlp);

    // Run initial forward pass
    mlp.forward([0, 0]);
    renderer.draw();
    boundary.draw();

    // --- Chart.js Setup ---
    const chartCtx = document.getElementById('mse-chart').getContext('2d');
    const mseData = {
        labels: [],
        datasets: [{
            label: 'Eav (Ortalama Karesel Hata)',
            data: [],
            borderColor: '#00e5ff',
            backgroundColor: 'rgba(0, 229, 255, 0.08)',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            fill: true,
            tension: 0.3
        }]
    };

    const mseChart = new Chart(chartCtx, {
        type: 'line',
        data: mseData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Epoch',
                        color: '#9090b8',
                        font: { family: 'Inter', size: 12 }
                    },
                    ticks: { color: '#606088', maxTicksLimit: 20, font: { size: 10 } },
                    grid: { color: 'rgba(42, 42, 90, 0.3)' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Eav',
                        color: '#9090b8',
                        font: { family: 'Inter', size: 12 }
                    },
                    ticks: { color: '#606088', font: { size: 10 } },
                    grid: { color: 'rgba(42, 42, 90, 0.3)' },
                    min: 0
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#9090b8',
                        font: { family: 'Inter', size: 11 }
                    }
                }
            }
        }
    });

    // --- UI References ---
    const lrSlider = document.getElementById('learning-rate');
    const lrValue = document.getElementById('lr-value');
    const momSlider = document.getElementById('momentum');
    const momValue = document.getElementById('mom-value');
    const activationSelect = document.getElementById('activation-fn');
    const speedSlider = document.getElementById('train-speed');
    const speedValue = document.getElementById('speed-value');
    const btnStep = document.getElementById('btn-step');
    const btnTrain = document.getElementById('btn-train');
    const btnStop = document.getElementById('btn-stop');
    const btnShuffle = document.getElementById('btn-shuffle');
    const btnReset = document.getElementById('btn-reset');
    const epochCount = document.getElementById('epoch-count');
    const mseValue = document.getElementById('mse-value');
    const mathBoard = document.getElementById('math-board');

    // --- Slider Events ---
    lrSlider.addEventListener('input', () => {
        mlp.learningRate = parseFloat(lrSlider.value);
        lrValue.textContent = parseFloat(lrSlider.value).toFixed(2);
    });

    momSlider.addEventListener('input', () => {
        mlp.momentum = parseFloat(momSlider.value);
        momValue.textContent = parseFloat(momSlider.value).toFixed(2);
    });

    activationSelect.addEventListener('change', () => {
        mlp.activationFn = activationSelect.value;
    });

    speedSlider.addEventListener('input', () => {
        epochsPerTick = parseInt(speedSlider.value);
        speedValue.textContent = epochsPerTick + 'x';
    });

    // --- Math Board ---
    function updateMathBoard(input, target) {
        if (!mathBoard) return;
        const state = mlp.getNetworkState();
        const fnName = mlp.activationFn === 'tanh' ? 'tanh' : 'σ';
        const x1 = input[0], x2 = input[1];

        let html = '<div class="mb-title">📐 Son Hesaplama Adımları</div>';
        html += `<div class="mb-label">Giriş: x₁=${x1}, x₂=${x2} | Hedef: d=${target[0]}</div>`;

        // Hidden layer
        if (state.nets[0] && state.outputs[1]) {
            html += '<div class="mb-section">▸ İleri Geçiş (Gizli Katman)</div>';
            for (let j = 0; j < mlp.layers[1]; j++) {
                const w1 = mlp.weights[0][j][0];
                const w2 = mlp.weights[0][j][1];
                const b = mlp.biases[0][j];
                const net = state.nets[0][j];
                const y = state.outputs[1][j];
                html += `<div class="mb-calc">net<sub>h${j+1}</sub> = <span class="mb-w">${w1.toFixed(2)}</span>·${x1} + <span class="mb-w">${w2.toFixed(2)}</span>·${x2} + <span class="mb-b">${b.toFixed(2)}</span> = <span class="mb-r">${net.toFixed(3)}</span></div>`;
                html += `<div class="mb-calc">y<sub>h${j+1}</sub> = ${fnName}(${net.toFixed(3)}) = <span class="mb-v">${y.toFixed(5)}</span></div>`;
            }
        }

        // Output layer
        if (state.nets[1] && state.outputs[2]) {
            html += '<div class="mb-section">▸ İleri Geçiş (Çıkış)</div>';
            const w1 = mlp.weights[1][0][0];
            const w2 = mlp.weights[1][0][1];
            const b = mlp.biases[1][0];
            const h1 = state.outputs[1][0];
            const h2 = state.outputs[1][1];
            const net = state.nets[1][0];
            const y = state.outputs[2][0];
            html += `<div class="mb-calc">net<sub>y</sub> = <span class="mb-w">${w1.toFixed(2)}</span>·<span class="mb-v">${h1.toFixed(3)}</span> + <span class="mb-w">${w2.toFixed(2)}</span>·<span class="mb-v">${h2.toFixed(3)}</span> + <span class="mb-b">${b.toFixed(2)}</span> = <span class="mb-r">${net.toFixed(3)}</span></div>`;
            html += `<div class="mb-calc">y = ${fnName}(${net.toFixed(3)}) = <span class="mb-v">${y.toFixed(5)}</span></div>`;
        }

        // Error & backprop
        if (state.errors.length > 0 && state.deltas.length > 0) {
            const e = state.errors[0];
            const y = state.outputs[2]?.[0] || 0;
            const phiPrime = mlp.activationDerivative(y);
            const deltaOut = state.deltas[1]?.[0] || 0;

            html += '<div class="mb-section">▸ Hata & Geri Yayılım</div>';
            html += `<div class="mb-calc"><span class="mb-e">e = d − y = ${target[0]} − ${y.toFixed(5)} = ${e.toFixed(5)}</span></div>`;
            html += `<div class="mb-calc">φ'(v) = ${phiPrime.toFixed(5)}</div>`;
            html += `<div class="mb-calc"><span class="mb-d">δ<sub>çıkış</sub> = e · φ'(v) = ${e.toFixed(4)} × ${phiPrime.toFixed(4)} = ${deltaOut.toFixed(6)}</span></div>`;

            // Hidden deltas
            if (state.deltas[0]) {
                for (let j = 0; j < mlp.layers[1]; j++) {
                    const dh = state.deltas[0][j];
                    html += `<div class="mb-calc"><span class="mb-d">δ<sub>h${j+1}</sub> = ${dh.toFixed(6)}</span></div>`;
                }
            }
        }

        mathBoard.innerHTML = html;
    }

    // --- Training Functions ---

    function runOneEpoch() {
        epoch++;
        const eav = mlp.trainEpoch(trainingData);

        // Update chart
        mseData.labels.push(epoch);
        mseData.datasets[0].data.push(eav);
        mseChart.update();

        epochCount.textContent = epoch;
        mseValue.textContent = eav.toFixed(6);

        updateOutputDisplay();
        highlightXorRow(-1);

        // Update math board with last sample
        const lastSample = trainingData[trainingData.length - 1];
        mlp.forward(lastSample.slice(0, 2));
        mlp.backward(lastSample.slice(2));
        updateMathBoard(lastSample.slice(0, 2), lastSample.slice(2));

        // Check convergence
        if (eav < 0.01 && !hasConverged) {
            hasConverged = true;
            showConvergenceCelebration();
        }

        return eav;
    }

    function updateOutputDisplay() {
        const pairs = [[0, 0], [0, 1], [1, 0], [1, 1]];
        const ids = ['out-00', 'out-01', 'out-10', 'out-11'];

        for (let i = 0; i < pairs.length; i++) {
            const output = mlp.forward(pairs[i]);
            const el = document.getElementById(ids[i]);
            el.textContent = output[0].toFixed(5);

            const target = XOR_DATA[i][2];
            const error = Math.abs(target - output[0]);
            if (error < 0.1) {
                el.style.color = '#22d3ae';
            } else if (error < 0.3) {
                el.style.color = '#ff9f43';
            } else {
                el.style.color = '#ff4e6a';
            }
        }
    }

    function highlightXorRow(idx) {
        document.querySelectorAll('.xor-row').forEach(r => r.classList.remove('active'));
        if (idx >= 0 && idx < 4) {
            document.querySelector(`.xor-row[data-idx="${idx}"]`)?.classList.add('active');
        }
    }

    // --- Convergence Celebration ---
    function showConvergenceCelebration() {
        const overlay = document.getElementById('convergence-overlay');
        if (overlay) {
            overlay.classList.add('show');
            // Create confetti
            for (let i = 0; i < 50; i++) {
                const confetti = document.createElement('div');
                confetti.className = 'confetti-piece';
                confetti.style.left = Math.random() * 100 + '%';
                confetti.style.animationDelay = Math.random() * 2 + 's';
                confetti.style.backgroundColor = ['#00e5ff', '#a855f7', '#22d3ae', '#ff9f43', '#ff4ecd', '#4ea8ff'][Math.floor(Math.random() * 6)];
                overlay.appendChild(confetti);
            }
            setTimeout(() => {
                overlay.classList.remove('show');
                overlay.querySelectorAll('.confetti-piece').forEach(c => c.remove());
            }, 4000);
        }
    }

    // --- Animated Step (1 Epoch with animation) ---

    function animatedStep() {
        btnStep.disabled = true;
        btnTrain.disabled = true;

        const sample = trainingData[0];
        const input = sample.slice(0, 2);
        const target = sample.slice(2);

        // Highlight XOR row
        const sampleIdx = XOR_DATA.findIndex(s => s[0] === input[0] && s[1] === input[1]);
        highlightXorRow(sampleIdx);

        // Forward pass
        mlp.forward(input);
        updateMathBoard(input, target);
        renderer.spawnForwardParticles();

        let phase = 'forward';

        function animLoop() {
            if (phase === 'forward') {
                const alive = renderer.updateParticles();
                renderer.draw();
                if (!alive) {
                    // Backward pass
                    mlp.backward(target);
                    updateMathBoard(input, target);
                    renderer.spawnBackwardParticles();
                    phase = 'backward';
                }
                animationFrameId = requestAnimationFrame(animLoop);
            } else if (phase === 'backward') {
                const alive = renderer.updateParticles();
                renderer.draw();
                if (!alive) {
                    renderer.clearParticles();
                    renderer.showGradientFlow = true;
                    mlp.updateWeights();

                    // Train remaining samples
                    for (let i = 1; i < trainingData.length; i++) {
                        const s = trainingData[i];
                        mlp.trainSample(s.slice(0, 2), s.slice(2));
                    }
                    epoch++;

                    // Calculate Eav
                    let totalErr = 0;
                    for (const s of trainingData) {
                        const out = mlp.forward(s.slice(0, 2));
                        const e = s[2] - out[0];
                        totalErr += 0.5 * e * e;
                    }
                    const eav = totalErr / trainingData.length;

                    mseData.labels.push(epoch);
                    mseData.datasets[0].data.push(eav);
                    mseChart.update();
                    epochCount.textContent = epoch;
                    mseValue.textContent = eav.toFixed(6);
                    updateOutputDisplay();
                    boundary.draw();
                    renderer.draw();

                    if (eav < 0.01 && !hasConverged) {
                        hasConverged = true;
                        showConvergenceCelebration();
                    }

                    btnStep.disabled = false;
                    btnTrain.disabled = false;
                    return;
                }
                animationFrameId = requestAnimationFrame(animLoop);
            }
        }

        animationFrameId = requestAnimationFrame(animLoop);
    }

    // --- Continuous Training ---

    function startContinuousTraining() {
        isTraining = true;
        btnTrain.disabled = true;
        btnStep.disabled = true;
        btnStop.disabled = false;

        trainInterval = setInterval(() => {
            for (let i = 0; i < epochsPerTick; i++) {
                runOneEpoch();
            }
            boundary.draw();
            renderer.draw();
        }, 50);
    }

    function stopTraining() {
        isTraining = false;
        if (trainInterval) {
            clearInterval(trainInterval);
            trainInterval = null;
        }
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        btnTrain.disabled = false;
        btnStep.disabled = false;
        btnStop.disabled = true;
        renderer.clearParticles();
        renderer.draw();
    }

    function resetNetwork() {
        stopTraining();
        mlp.initWeights();
        epoch = 0;
        hasConverged = false;
        trainingData = [...XOR_DATA];
        renderer.showGradientFlow = false;

        mseData.labels = [];
        mseData.datasets[0].data = [];
        mseChart.update();

        epochCount.textContent = '0';
        mseValue.textContent = '-';

        ['out-00', 'out-01', 'out-10', 'out-11'].forEach(id => {
            const el = document.getElementById(id);
            el.textContent = '-';
            el.style.color = '#00e5ff';
        });

        if (mathBoard) mathBoard.innerHTML = '<div class="mb-title">📐 Hesaplama Tahtası</div><div class="mb-label">Eğitim başladığında burada canlı formüller görünecek.</div>';

        mlp.forward([0, 0]);
        renderer.draw();
        boundary.draw();
    }

    function shuffleData() {
        for (let i = trainingData.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [trainingData[i], trainingData[j]] = [trainingData[j], trainingData[i]];
        }
        btnShuffle.style.boxShadow = '0 0 20px rgba(168, 85, 247, 0.5)';
        setTimeout(() => { btnShuffle.style.boxShadow = ''; }, 300);
    }

    // --- Button Events ---
    btnStep.addEventListener('click', animatedStep);
    btnTrain.addEventListener('click', startContinuousTraining);
    btnStop.addEventListener('click', stopTraining);
    btnShuffle.addEventListener('click', shuffleData);
    btnReset.addEventListener('click', resetNetwork);

    // --- Idle animation loop ---
    function idleRender() {
        if (!isTraining) {
            renderer.draw();
        }
        requestAnimationFrame(idleRender);
    }
    requestAnimationFrame(idleRender);
});
