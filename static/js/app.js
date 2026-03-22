/* ========================================
   app.js - Main Application Controller
   Ties together MLP engine, Canvas renderer,
   Chart.js MSE graph, and UI controls.
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

    // --- Initialize MLP & Renderer ---
    const mlp = new MLP();
    const renderer = new CanvasRenderer('network-canvas', mlp);

    // Run initial forward pass so neurons have values to display
    mlp.forward([0, 0]);
    renderer.draw();

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
    const btnStep = document.getElementById('btn-step');
    const btnTrain = document.getElementById('btn-train');
    const btnStop = document.getElementById('btn-stop');
    const btnShuffle = document.getElementById('btn-shuffle');
    const btnReset = document.getElementById('btn-reset');
    const epochCount = document.getElementById('epoch-count');
    const mseValue = document.getElementById('mse-value');

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

    // --- Training Functions ---

    function runOneEpoch() {
        epoch++;
        const eav = mlp.trainEpoch(trainingData);

        // Update chart
        mseData.labels.push(epoch);
        mseData.datasets[0].data.push(eav);
        mseChart.update();

        // Update stats
        epochCount.textContent = epoch;
        mseValue.textContent = eav.toFixed(6);

        // Update network outputs display
        updateOutputDisplay();

        // Highlight active XOR row (last one trained)
        highlightXorRow(-1); // clear

        return eav;
    }

    function updateOutputDisplay() {
        const pairs = [[0, 0], [0, 1], [1, 0], [1, 1]];
        const ids = ['out-00', 'out-01', 'out-10', 'out-11'];

        for (let i = 0; i < pairs.length; i++) {
            const output = mlp.forward(pairs[i]);
            const el = document.getElementById(ids[i]);
            el.textContent = output[0].toFixed(5);

            // Color based on correctness
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

    // --- Animated Step (1 Epoch with animation) ---

    function animatedStep() {
        btnStep.disabled = true;
        btnTrain.disabled = true;

        const sample = trainingData[0];
        const input = sample.slice(0, 2);
        const target = sample.slice(2);

        // Forward pass
        mlp.forward(input);
        renderer.spawnForwardParticles();

        let phase = 'forward';

        function animLoop() {
            if (phase === 'forward') {
                const alive = renderer.updateParticles();
                renderer.draw();
                if (!alive) {
                    // Backward pass
                    mlp.backward(target);
                    renderer.spawnBackwardParticles();
                    phase = 'backward';
                }
                animationFrameId = requestAnimationFrame(animLoop);
            } else if (phase === 'backward') {
                const alive = renderer.updateParticles();
                renderer.draw();
                if (!alive) {
                    // Update weights and run full epoch
                    renderer.clearParticles();
                    mlp.updateWeights();
                    // Now run the rest of the epoch
                    for (let i = 1; i < trainingData.length; i++) {
                        const s = trainingData[i];
                        mlp.trainSample(s.slice(0, 2), s.slice(2));
                    }
                    epoch++;
                    // Calculate Eav for this epoch
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
                    renderer.draw();

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
            // Run multiple epochs per tick for speed
            for (let i = 0; i < 10; i++) {
                runOneEpoch();
            }
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
        trainingData = [...XOR_DATA];

        // Clear chart
        mseData.labels = [];
        mseData.datasets[0].data = [];
        mseChart.update();

        epochCount.textContent = '0';
        mseValue.textContent = '-';

        // Reset outputs
        ['out-00', 'out-01', 'out-10', 'out-11'].forEach(id => {
            const el = document.getElementById(id);
            el.textContent = '-';
            el.style.color = '#00e5ff';
        });

        mlp.forward([0, 0]);
        renderer.draw();
    }

    function shuffleData() {
        // Fisher-Yates shuffle
        for (let i = trainingData.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [trainingData[i], trainingData[j]] = [trainingData[j], trainingData[i]];
        }
        // Flash shuffle button
        btnShuffle.style.boxShadow = '0 0 20px rgba(168, 85, 247, 0.5)';
        setTimeout(() => { btnShuffle.style.boxShadow = ''; }, 300);
    }

    // --- Button Events ---
    btnStep.addEventListener('click', animatedStep);
    btnTrain.addEventListener('click', startContinuousTraining);
    btnStop.addEventListener('click', stopTraining);
    btnShuffle.addEventListener('click', shuffleData);
    btnReset.addEventListener('click', resetNetwork);

    // --- Idle animation loop (always render canvas) ---
    function idleRender() {
        if (!isTraining) {
            renderer.draw();
        }
        requestAnimationFrame(idleRender);
    }
    requestAnimationFrame(idleRender);
});
