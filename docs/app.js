// Espera a que el DOM esté completamente cargado
document.addEventListener('DOMContentLoaded', () => {

    // --- Conexión WebSocket ---
    // Se conecta automáticamente al servidor Flask
    const socket = io("https://ia-proyecto-b1-redes-neuronales-amca.onrender.com");

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
            const response = await fetch('https://ia-proyecto-b1-redes-neuronales-amca.onrender.com/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x, y, z })
            });

            if (!response.ok) {
                throw new Error('Error en la respuesta del servidor.');
            }

            const data = await response.json(); // ej: {"clase": 0, "probabilidad": 0.0123}
            
            // --- CAMBIO AQUÍ ---
            const claseStr = data.clase === 1 ? 'Clase 1 (Rojo)' : 'Clase 0 (Amarillo)';
            const probStr = `(Prob: ${data.probabilidad.toFixed(4)})`;

            resultLabel.innerHTML = `${claseStr} <br> ${probStr}`;
            // También usamos 'alert-danger' para rojo y 'alert-warning' para amarillo
            resultLabel.className = data.clase === 1 ? 'alert alert-danger text-center' : 'alert alert-warning text-center';
            // --- FIN DEL CAMBIO ---

            // Actualizar la gráfica 3D mostrando el nuevo punto
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
        btnTrain.disabled = true;
        btnTrain.textContent = 'Entrenando...';
        trainStatus.textContent = 'Iniciando entrenamiento...';
        progressBar.style.width = '0%';
        progressBar.classList.add('progress-bar-animated');

        // Inicializar la gráfica de loss vacía
        Plotly.newPlot('loss-chart', [{ x: [], y: [], mode: 'lines', name: 'Loss' }], layoutLoss);

        // Enviar la orden de entrenar al backend
        socket.emit('start_training');
    }

    /**
     * 3. ACTUALIZAR GRÁFICA 3D: Llama a la API /api/plot_data
     * @param {object} [newPointData] - Opcional. Un objeto { newPoint: [x,y,z], newClass: 0|1 }
     */
    async function updatePlot3D(newPointData = null) {
        try {
            const response = await fetch('https://ia-proyecto-b1-redes-neuronales-amca.onrender.com/api/plot_data');
            if (!response.ok) throw new Error('No se pudieron cargar los datos de la gráfica.');

            const data = await response.json(); // ej: {"region0": [[...]], "region1": [[...]], ...}

            // --- Transformar datos JSON a traces de Plotly ---
            
            // Función auxiliar para 'unzip' las coordenadas
            const unzip = arr => arr[0] ? [arr.map(p => p[0]), arr.map(p => p[1]), arr.map(p => p[2])] : [[], [], []];

            const [r0_x, r0_y, r0_z] = unzip(data.region0);
            const [r1_x, r1_y, r1_z] = unzip(data.region1);
            const [m0_x, m0_y, m0_z] = unzip(data.muestras0);
            const [m1_x, m1_y, m1_z] = unzip(data.muestras1);

            // --- CAMBIO AQUÍ (en las propiedades 'name') ---
            const traces = [
                // Trace 0: Región Amarillo (0)
                { x: r0_x, y: r0_y, z: r0_z, mode: 'markers', type: 'scatter3d', name: 'Región Amarillo (0)', marker: { color: '#FFDDC1', size: 2, opacity: 0.2 } },
                // Trace 1: Región Rojo (1)
                { x: r1_x, y: r1_y, z: r1_z, mode: 'markers', type: 'scatter3d', name: 'Región Rojo (1)', marker: { color: '#FF9AA2', size: 2, opacity: 0.2 } },
                // Trace 2: Muestras Amarillo (0)
                { x: m0_x, y: m0_y, z: m0_z, mode: 'markers', type: 'scatter3d', name: 'Muestra Amarillo (0)', marker: { color: '#FFDDC1', size: 6, symbol: 'circle', line: { color: 'black', width: 1 } } },
                // Trace 3: Muestras Rojo (1)
                { x: m1_x, y: m1_y, z: m1_z, mode: 'markers', type: 'scatter3d', name: 'Muestra Rojo (1)', marker: { color: '#FF9AA2', size: 6, symbol: 'square', line: { color: 'black', width: 1 } } }
            ];
            // --- FIN DEL CAMBIO ---


            // Si hay un punto nuevo de predicción, añadirlo como un trace separado
            if (newPointData) {
                const [px, py, pz] = newPointData.newPoint;
                traces.push({
                    x: [px], y: [py], z: [pz],
                    mode: 'markers', type: 'scatter3d',
                    name: 'Predicción',
                    marker: {
                        color: newPointData.newClass === 1 ? '#7D3C98' : '#1ABC9C', // Morado o Verde
                        size: 8,
                        symbol: 'cross',
                        line: { color: 'black', width: 2 }
                    }
                });
            }

            // Dibujar la gráfica
            Plotly.newPlot('plot-3d', traces, layout3D, { responsive: true });

        } catch (error) {
            console.error('Error al actualizar gráfica 3D:', error);
            document.getElementById('plot-3d').innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
        }
    }

    // --- Handlers de WebSocket ---

    // Se recibe cuando el entrenamiento envía progreso
    socket.on('training_progress', (data) => {
        // data = { epoch, total_epochs, loss, pct, losses }
        progressBar.style.width = `${data.pct}%`;
        trainStatus.textContent = `Epoch ${data.epoch} / ${data.total_epochs} | Loss: ${data.loss.toFixed(6)}`;
        
        // Actualizar gráfica de Loss en tiempo real
        // 'losses' es el historial completo, lo reemplazamos
        Plotly.react('loss-chart', [{ 
            x: Array.from({ length: data.losses.length }, (_, i) => i * (data.total_epochs / data.losses.length)), // Escala X
            y: data.losses, 
            mode: 'lines', 
            name: 'Loss' 
        }], layoutLoss);
    });

    // Se recibe cuando el entrenamiento termina
    socket.on('training_finished', (data) => {
        btnTrain.disabled = false;
        btnTrain.textContent = 'Iniciar Entrenamiento';
        trainStatus.textContent = `¡Entrenamiento finalizado! Loss final: ${data.final_loss.toFixed(6)}`;
        progressBar.style.width = '100%';
        progressBar.classList.remove('progress-bar-animated');
        
        // Actualizar la gráfica 3D con el nuevo modelo entrenado
        updatePlot3D();
    });

    // Se recibe si hay un error en el entrenamiento
    socket.on('training_error', (data) => {
        btnTrain.disabled = false;
        btnTrain.textContent = 'Iniciar Entrenamiento';
        trainStatus.textContent = `Error: ${data.error}`;
        progressBar.style.width = '0%';
        progressBar.classList.remove('progress-bar-animated');
    });

    // --- Asignación de Eventos ---
    btnPredict.addEventListener('click', handlePredict);
    btnTrain.addEventListener('click', handleTrain);
    btnUpdatePlot.addEventListener('click', () => updatePlot3D());

    // --- Carga Inicial ---
    // Al cargar la página, dibuja la gráfica 3D con el modelo por defecto
    updatePlot3D();
});