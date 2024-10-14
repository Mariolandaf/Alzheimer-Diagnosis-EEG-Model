% Cargar la biblioteca de estadísticas de MATLAB
addpath(fullfile(matlabroot,'toolbox','stats'));

% Definir rutas de directorios
eeglab_dir = 'C:\Users\mario\Downloads\EGGLAB';

% Verificar existencia de directorios y archivos
assert(isfolder(eeglab_dir), 'El directorio EEGLAB no existe: %s', eeglab_dir);

% Añadir EEGLAB al path de MATLAB
addpath(eeglab_dir);

% Inicia EEGLAB
eeglab;

% Define las rutas a las carpetas
carpetas = {'C:\Dataset\A', 'C:\Dataset\sanos', 'C:\Dataset\FrontotemporalDementia'};

% Carga los archivos de cada carpeta
ALLEEG = [];
for i = 1:length(carpetas)
    ALLEEG = cargar_datos(carpetas{i}, ALLEEG);
end

%% Eliminar artefactos usando ASR e ICA
% Define los parámetros para ASR
window_len = 0.5; % Longitud de la ventana en segundos
window_SD = 20;  % Desviación estándar máxima aceptable de la ventana

% Aplica ASR e ICA a todos los conjuntos de datos en ALLEEG
for i = 1:length(ALLEEG)
    EEG = ALLEEG(i);
    
    % Visualiza los datos antes de la eliminación de artefactos para la primera encefalografía
    if i == 1
        pop_eegplot(EEG, 1, 1, 1);
    end
    
    % Aplica ASR
    try
        EEG = clean_artifacts(EEG, 'ChannelCriterion', 'off', 'LineNoiseCriterion', 'off', 'BurstCriterion', window_SD);
        % Verificar que los datos no estén vacíos después de ASR
        if isempty(EEG.data)
            error('Error: El dataset EEG está vacío después de ASR.');
        end
    catch ME
        warning('Error durante la aplicación de ASR en el dataset %d: %s', i, ME.message);
        continue; % Saltar al siguiente conjunto de datos
    end
    
    % Aplica ICA
    try
        EEG = pop_runica(EEG, 'extended', 1);
        % Verificar que los datos no estén vacíos después de ICA
        if isempty(EEG.data)
            error('Error: El dataset EEG está vacío después de ICA.');
        end
    catch ME
        warning('Error durante la aplicación de ICA en el dataset %d: %s', i, ME.message);
        continue; % Saltar al siguiente conjunto de datos
    end
    
    % Eliminar componentes ICA manualmente
    try
        pop_selectcomps(EEG, 1:EEG.nbchan);
        % Inspeccionar y eliminar componentes relacionados con artefactos manualmente
        % Esperar hasta que se completen las selecciones de componentes
        disp('Elimina manualmente los componentes relacionados con artefactos y presiona Enter para continuar...');
        pause;
        
        % Verificar que los datos no estén vacíos después de eliminar componentes
        if isempty(EEG.data)
            error('Error: El dataset EEG está vacío después de eliminar componentes ICA.');
        end
    catch ME
        warning('Error durante la eliminación de componentes ICA en el dataset %d: %s', i, ME.message);
        continue; % Saltar al siguiente conjunto de datos
    end
    
    % Visualiza los datos después de la eliminación de artefactos para la primera encefalografía
    if i == 1
        pop_eegplot(EEG, 1, 1, 1);
    end
    
    % Almacenar el dataset modificado en ALLEEG
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, i);
end

%% Agregar eventos
% Define los nombres y los tiempos de los eventos
eventos = {'Evento1', 'Evento2', 'Evento3'};
tiempos_eventos = [1, 2, 3]; % Tiempos de los eventos en segundos

% Verificar que los tiempos de eventos no sean vacíos y coincidan con los nombres de los eventos
if isempty(eventos) || isempty(tiempos_eventos) || length(eventos) ~= length(tiempos_eventos)
    error('Error: Los nombres y tiempos de los eventos deben estar definidos y tener la misma longitud.');
end

% Añade eventos a cada conjunto de datos en ALLEEG
for i = 1:length(ALLEEG)
    EEG = ALLEEG(i);
    
    % Añade los eventos uno por uno
    for j = 1:length(eventos)
        try
            % Calcular la latencia en muestras y verificar que no exceda la longitud de los datos
            latencia_muestras = EEG.srate * tiempos_eventos(j);
            if latencia_muestras > EEG.pnts
                warning('Advertencia: La latencia del evento %s excede la longitud de los datos y será omitido.', eventos{j});
                continue; % Saltar este evento
            end
            
            EEG.event(end+1).type = eventos{j}; % Tipo de evento
            EEG.event(end).latency = latencia_muestras; % Tiempo de ocurrencia del evento en muestras
            EEG.event(end).urevent = length(EEG.event); % Número de evento único
        catch ME
            warning('Error al añadir el evento %s al dataset %d: %s', eventos{j}, i, ME.message);
            continue; % Saltar al siguiente evento
        end
    end
    
    % Ordena los eventos por latencia
    try
        EEG = eeg_checkset(EEG, 'eventconsistency');
    catch ME
        warning('Error al ordenar los eventos en el dataset %d: %s', i, ME.message);
        continue; % Saltar al siguiente conjunto de datos
    end
    
    % Almacena el conjunto de datos modificado en la estructura ALLEEG
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, i);
end

% Mensaje informativo
fprintf('Se han agregado eventos a todos los conjuntos de datos en ALLEEG.\n');


%% Filtro de Banda
% Define las frecuencias de corte para el filtro
low_cutoff = 0.5; % Frecuencia de corte baja en Hz
high_cutoff = 50; % Frecuencia de corte alta en Hz

% Aplica el filtro a todos los conjuntos de datos en ALLEEG
for i = 1:length(ALLEEG)
    EEG = ALLEEG(i);
    
    if i == 1
        % Visualiza los datos antes de la aplicación del filtro para la primera encefalografía
        pop_eegplot(EEG, 1, 1, 1);
    end
    
    % Diseña el filtro
    d = designfilt('bandpassiir', 'FilterOrder', 10, ...
        'HalfPowerFrequency1', low_cutoff, 'HalfPowerFrequency2', high_cutoff, ...
        'SampleRate', EEG.srate);
    
    % Aplica el filtro
    for ch = 1:EEG.nbchan
        EEG.data(ch, :) = filtfilt(d, double(EEG.data(ch, :)));
    end
    
    if i == 1
        % Visualiza los datos después de la aplicación del filtro para la primera encefalografía
        pop_eegplot(EEG, 1, 1, 1);
    end
    
    % Almacena el conjunto de datos modificado en la estructura ALLEEG
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, i);
end


%% Re-referenciación
% Realiza la re-referenciación para cada conjunto de datos en ALLEEG
for i = 1:length(ALLEEG)
    EEG = ALLEEG(i);
    
    % Realiza la re-referenciación
    EEG = pop_reref(EEG, [], 'keepref', 'on', 'exclude', []);

    % Guarda el resultado de la re-referenciación
    ALLEEG(i) = EEG;

    % Almacena el conjunto de datos re-referenciado en la estructura ALLEEG
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, i);
    
    % Muestra los resultados solo para la primera encefalografía
    if i == 1
        % Visualiza los datos después de la re-referenciación para la primera encefalografía
        pop_eegplot(EEG, 1, 1, 1);
    end
end


%% Recorte de épocas
% Define los parámetros para el recorte de épocas
tiempo_inicio = 0;  % Tiempo de inicio de la época en segundos (antes del estímulo)
tiempo_final = 4;   % Tiempo final de la época en segundos (después del estímulo)

% Recorta épocas para cada conjunto de datos en ALLEEG
for i = 1:length(ALLEEG)
    EEG = ALLEEG(i);
    
    % Verifica si hay eventos en el intervalo de tiempo especificado
    evento_en_rango = false;
    for j = 1:length(EEG.event)
        if EEG.event(j).latency / EEG.srate >= tiempo_inicio && EEG.event(j).latency / EEG.srate <= tiempo_final
            evento_en_rango = true;
            break;
        end
    end
    
    if evento_en_rango
        % Recorta épocas basadas en el intervalo de tiempo especificado
        try
            EEG = pop_epoch(EEG, {}, [tiempo_inicio tiempo_final]);
            
            % Verificar que los datos no estén vacíos después del recorte de épocas
            if isempty(EEG.data)
                error('Error: El dataset EEG está vacío después del recorte de épocas.');
            end
            
            % Guarda el resultado del recorte de épocas
            ALLEEG(i) = EEG;
            
            % Muestra los resultados solo para la primera encefalografía
            if i == 1
                % Visualiza los datos después del recorte de épocas para la primera encefalografía
                pop_eegplot(EEG, 1, 1, 1);
            end
        catch ME
            warning('Error durante el recorte de épocas en el dataset %d: %s', i, ME.message);
            continue; % Saltar al siguiente conjunto de datos
        end
    else
        warning('No hay eventos en el rango de tiempo especificado para el dataset %d.', i);
        continue; % Saltar al siguiente conjunto de datos
    end
end

% Mensaje informativo
fprintf('Se han recortado las épocas de todos los conjuntos de datos en ALLEEG.\n');


%% Promedio de épocas
% Define los parámetros para el promedio de épocas
tiempo_inicio = 0;  % Tiempo de inicio de la época en segundos (antes del estímulo)
tiempo_final = 4;   % Tiempo final de la época en segundos (después del estímulo)

% Promedia las épocas para cada conjunto de datos en ALLEEG
for i = 1:length(ALLEEG)
    EEG = ALLEEG(i);
    
    % Verifica si hay eventos en el intervalo de tiempo especificado
    evento_en_rango = false;
    for j = 1:length(EEG.event)
        if EEG.event(j).latency / EEG.srate >= tiempo_inicio && EEG.event(j).latency / EEG.srate <= tiempo_final
            evento_en_rango = true;
            break;
        end
    end
    
    if evento_en_rango
        try
            % Recorta épocas basadas en el intervalo de tiempo especificado
            EEG = pop_epoch(EEG, {}, [tiempo_inicio tiempo_final]);

            % Verificar que los datos no estén vacíos después del recorte de épocas
            if isempty(EEG.data)
                warning('El dataset EEG está vacío después del recorte de épocas para el conjunto %d.', i);
                continue; % Saltar al siguiente conjunto de datos
            end

            % Promedia las épocas para cada canal y cada condición
            EEG = pop_rmdat(EEG, {}, [tiempo_inicio tiempo_final]);

            % Guarda el resultado del promedio de épocas
            ALLEEG(i) = EEG;

            % Muestra los resultados solo para la primera encefalografía
            if i == 1
                % Visualiza los datos después del promedio de épocas para la primera encefalografía
                pop_eegplot(EEG, 1, 1, 1);
            end
        catch ME
            warning('Error durante el promedio de épocas en el dataset %d: %s', i, ME.message);
            continue; % Saltar al siguiente conjunto de datos
        end
    else
        warning('No hay eventos en el rango de tiempo especificado para el dataset %d.', i);
        continue; % Saltar al siguiente conjunto de datos
    end
end

% Mensaje informativo
fprintf('Se han promediado las épocas de todos los conjuntos de datos en ALLEEG.\n');

%% Normalización de datos
% Normaliza los datos después del recorte de épocas
for i = 1:length(ALLEEG)
    EEG = ALLEEG(i);
    
    % Normaliza los datos utilizando Z-score
    EEG.data = zscore_custom(EEG.data);
    
    % Muestra los resultados solo para la primera encefalografía después de la normalización
    if i == 1
        % Visualiza los datos después de la normalización para la primera encefalografía
        pop_eegplot(EEG, 1, 1, 1);
    end
    
    % Guarda el resultado de la normalización en la estructura ALLEEG
    ALLEEG(i) = EEG;
end

%% Interpolación de electrodos faltantes
for i = 1:length(ALLEEG)
    EEG = ALLEEG(i);
    
    % Interpola los electrodos faltantes
    EEG = pop_interp(EEG, EEG.chanlocs, 'spherical');
    if i == 1
        % Visualiza los datos después de la normalización para la primera encefalografía
        pop_eegplot(EEG, 1, 1, 1);
    end
    % Guarda el resultado de la interpolación en la estructura ALLEEG
    ALLEEG(i) = EEG;
end
%% Eliminar épocas con artefactos faltantes
% Define los parámetros para la eliminación de épocas con artefactos faltantes
% (si es necesario)

% Aplica la eliminación de épocas con artefactos faltantes a cada conjunto de datos en ALLEEG
for i = 1:length(ALLEEG)
    EEG = ALLEEG(i);
    
    % Aplica la eliminación de épocas con artefactos faltantes
    % (inserta aquí el código específico para esta etapa)
    
    % Muestra los resultados solo para la primera encefalografía
    if i == 1
        % Muestra el resultado de la primera encefalografía después de eliminar las épocas con artefactos faltantes
        pop_eegplot(EEG, 1, 1, 1);
    end
    
    % Guarda el resultado de la eliminación de épocas con artefactos faltantes
    ALLEEG(i) = EEG;
end

%% Exportar datos


% Exportar cada conjunto de datos preprocesado en archivos individuales
for i = 1:length(ALLEEG)
    EEG = ALLEEG(i);
    datos_preprocesados = struct();
    datos_preprocesados.data = EEG.data;
    datos_preprocesados.fs = EEG.srate;
    datos_preprocesados.eventos = {EEG.event.type};
    datos_preprocesados.paciente = EEG.setname;
    
    % Define la ruta del archivo de salida
    nombre_archivo = sprintf('datos_preprocesados_%d.mat', i);
    ruta_archivo = fullfile('C:\Dataset', nombre_archivo); % Cambia la ruta según sea necesario
    
    % Exporta los datos preprocesados en un archivo .mat
    exportar_datos_preprocesados(datos_preprocesados, ruta_archivo);
end



%% funciones

% Definición de la función cargar_datos
function ALLEEG = cargar_datos(carpeta, ALLEEG)
    archivos = dir(fullfile(carpeta, '*.set'));
    for i = 1:length(archivos)
        EEG = pop_loadset('filename', archivos(i).name, 'filepath', carpeta);
        [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
    end
end

function data_normalized = zscore_custom(data)
    % Calcula la media y la desviación estándar de los datos
    mu = mean(data, 2);     % Media a lo largo de las columnas
    sigma = std(data, 0, 2);  % Desviación estándar a lo largo de las columnas
    
    % Normaliza los datos usando Z-score
    data_normalized = (data - mu) ./ sigma;
end

% Exportar datos preprocesados en un formato compatible con Python
function exportar_datos_preprocesados(datos, ruta_archivo)
    % Guarda los datos en un archivo .mat
    save(ruta_archivo, 'datos');
end
