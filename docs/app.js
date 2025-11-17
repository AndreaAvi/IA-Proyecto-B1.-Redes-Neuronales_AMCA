// Espera a que el DOM esté completamente cargado
document.addEventListener('DOMContentLoaded', () => {

    // --- Conexión WebSocket ---
    // URL de nuestro backend en Render
    const BACKEND_URL = "https://ia-proyecto-b1-redes-neuronales-amca.onrender.com"; 

    // Se conecta explícitamente al servidor de Render
    const socket = io(BACKEND_URL);


    socket.on('connect', () => {
        console.log('Conectado al servidor WebSocket');
    });

    socket.on('disconnect', () => {
        console.log('Desconectado del servidor WebSocket');
    });

    // --- Referencias a elementos del DOM ---
    const btnPredict = document.getElementById('btn-predict');
    const btnTrain = document.getElementById('btn-train');
    const btnUpdatePlot = document.getElementById('btn-update-plot');
    
    const entryX = document.getElementById('entry-x');
    const entryY = document.getElementById('entry-y');
    const entryZ = document.getElementById('entry-z');
    
    const resultLabel = document.getElementById('result-label');
    const progressBar = document.getElementById('train-progress-bar');
    const trainStatus = document.getElementById('train-status');
    const loadingOverlay = document.getElementById('loading-overlay'); // <-- Referencia al overlay

    // --- Layouts de Gráficas (Plotly) ---
    const layout3D = {
        title: 'Regiones de Decisión y Muestras 3D',
        scene: {
            xaxis: { title: 'X' },
            yaxis: { title: 'Y' },
            zaxis: { title: 'Z' }
        },
        margin: { l: 0, r: 0, b: 0, t: 40 } // Ajuste de márgenes
    };

    const layoutLoss = {
        title: 'Curva de Loss (Entrenamiento)',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Loss (Binary Cross-Entropy)' },
        margin: { l: 50, r: 20, b: 40, t: 40 }
    };

    // --- Funciones de API ---

    /**
     * 1. PREDECIR: Llama a la API /api/predict
     */
    async function handlePredict() {
        const x = parseFloat(entryX.value);
        const y = parseFloat(entryY.value);
        const z = parseFloat(entryZ.value);

        if (isNaN(x) || isNaN(y) || isNaN(z)) {
            resultLabel.textContent = 'Error: Ingresa números válidos.';
            resultLabel.className = 'alert alert-danger text-center';
            return;
        }

        try {
            const response = await fetch(BACKEND_URL + '/api/predict', { // <-- URL CORREGIDA
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x, y, z })
            });

            if (!response.ok) {
                throw new Error('Error en la respuesta del servidor.');
            }

            const data = await response.json(); 
            
            const claseStr = data.clase === 1 ? 'Clase 1 (Rojo)' : 'Clase 0 (Amarillo)';
            const probStr = `(Prob: ${data.probabilidad.toFixed(4)})`;

            resultLabel.innerHTML = `${claseStr} <br> ${probStr}`;
            resultLabel.className = data.clase === 1 ? 'alert alert-danger text-center' : 'alert alert-warning text-center';

            updatePlot3D({ newPoint: [x, y, z], newClass: data.clase });

        } catch (error) {
            console.error('Error en predicción:', error);
            resultLabel.textContent = `Error: ${error.message}`;
            resultLabel.className = 'alert alert-danger text-center';
        }
    }

    /**
     * 2. ENTRENAR: Envía evento 'start_training' por WebSocket
     */
    function handleTrain() {
        // --- CAMBIO 1 (Comprobación) ---
        if (loadingOverlay) loadingOverlay.style.display = 'flex';
        if (btnTrain) btnTrain.disabled = true;
        if (btnPredict) btnPredict.disabled = true;
        if (btnUpdatePlot) btnUpdatePlot.disabled = true;

        if (btnTrain) btnTrain.textContent = 'Entrenando...';
        if (trainStatus) trainStatus.textContent = 'Iniciando entrenamiento...';
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.classList.add('progress-bar-animated');
        }

        Plotly.newPlot('loss-chart', [{ x: [], y: [], mode: 'lines', name: 'Loss' }], layoutLoss);

        socket.emit('start_training');
    }

    /**
     * 3. ACTUALIZAR GRÁFICA 3D: Llama a la API /api/plot_data
     * @param {object} [newPointData] - Opcional. Un objeto { newPoint: [x,y,z], newClass: 0|1 }
     */
    async function updatePlot3D(newPointData = null) {
        try {
            const response = await fetch(BACKEND_URL + '/api/plot_data'); // <-- URL CORREGIDA
            if (!response.ok) throw new Error('No se pudieron cargar los datos de la gráfica.');

            const data = await response.json(); 

            const unzip = arr => arr[0] ? [arr.map(p => p[0]), arr.map(p => p[1]), arr.map(p => p[2])] : [[], [], []];

            const [r0_x, r0_y, r0_z] = unzip(data.region0);
            const [r1_x, r1_y, r1_z] = unzip(data.region1);
            const [m0_x, m0_y, m0_z] = unzip(data.muestras0);
            const [m1_x, m1_y, m1_z] = unzip(data.muestras1);

            const traces = [
                { x: r0_x, y: r0_y, z: r0_z, mode: 'markers', type: 'scatter3d', name: 'Región Amarillo (0)', marker: { color: '#FFDDC1', size: 2, opacity: 0.2 } },
                { x: r1_x, y: r1_y, z: r1_z, mode: 'markers', type: 'scatter3d', name: 'Región Rojo (1)', marker: { color: '#FF9AA2', size: 2, opacity: 0.2 } },
                { x: m0_x, y: m0_y, z: m0_z, mode: 'markers', type: 'scatter3d', name: 'Muestra Amarillo (0)', marker: { color: '#FFDDC1', size: 6, symbol: 'circle', line: { color: 'black', width: 1 } } },
                { x: m1_x, y: m1_y, z: m1_z, mode: 'markers', type: 'scatter3d', name: 'Muestra Rojo (1)', marker: { color: '#FF9AA2', size: 6, symbol: 'square', line: { color: 'black', width: 1 } } }
            ];

            if (newPointData) {
                const [px, py, pz] = newPointData.newPoint;
                traces.push({
                    x: [px], y: [py], z: [pz],
                    mode: 'markers', type: 'scatter3d',
                    name: 'Predicción',
                    marker: {
                        color: newPointData.newClass === 1 ? '#7D3C98' : '#1ABC9C',
                        size: 8,
                        symbol: 'cross',
                        line: { color: 'black', width: 2 }
                    }
                });
            }

            Plotly.newPlot('plot-3d', traces, layout3D, { responsive: true });

        } catch (error) {
            console.error('Error al actualizar gráfica 3D:', error);
            document.getElementById('plot-3d').innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
        }
    }

    // --- Handlers de WebSocket ---

    socket.on('training_progress', (data) => {
        if (progressBar) progressBar.style.width = `${data.pct}%`;
        if (trainStatus) trainStatus.textContent = `Epoch ${data.epoch} / ${data.total_epochs} | Loss: ${data.loss.toFixed(6)}`;
        
        Plotly.react('loss-chart', [{ 
            x: Array.from({ length: data.losses.length }, (_, i) => i * (data.total_epochs / data.losses.length)),
            y: data.losses, 
            mode: 'lines', 
            name: 'Loss' 
        }], layoutLoss);
    });

    socket.on('training_finished', (data) => {
        // --- CAMBIO 2 (Comprobación) ---
        if (loadingOverlay) loadingOverlay.style.display = 'none';
        if (btnTrain) btnTrain.disabled = false;
        if (btnPredict) btnPredict.disabled = false;
        if (btnUpdatePlot) btnUpdatePlot.disabled = false;
        
        if (btnTrain) btnTrain.textContent = 'Iniciar Entrenamiento';
        if (trainStatus) trainStatus.textContent = `¡Entrenamiento finalizado! Loss final: ${data.final_loss.toFixed(6)}`;
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.classList.remove('progress-bar-animated');
        }
        
        updatePlot3D();
    });

    socket.on('training_error', (data) => {
        // --- CAMBIO 3 (Comprobación) ---
        if (loadingOverlay) loadingOverlay.style.display = 'none';
        if (btnTrain) btnTrain.disabled = false;
        if (btnPredict) btnPredict.disabled = false;
        if (btnUpdatePlot) btnUpdatePlot.disabled = false;

        if (btnTrain) btnTrain.textContent = 'Iniciar Entrenamiento';
        if (trainStatus) trainStatus.textContent = `Error: ${data.error}`;
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.classList.remove('progress-bar-animated');
        }
    });

    // --- Asignación de Eventos ---
    // --- CAMBIOS 4, 5, 6 (Comprobación) ---
    if (btnPredict) btnPredict.addEventListener('click', handlePredict);
    if (btnTrain) btnTrain.addEventListener('click', handleTrain);
    if (btnUpdatePlot) btnUpdatePlot.addEventListener('click', () => updatePlot3D());

    // --- Carga Inicial ---
    updatePlot3D();
});