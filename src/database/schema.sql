-- Crear la base de datos
CREATE DATABASE IF NOT EXISTS accidentes_db;
USE accidentes_db;

-- Tabla de accidentes
CREATE TABLE IF NOT EXISTS accidentes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fecha DATE NOT NULL,
    hora TIME NOT NULL,
    ubicacion VARCHAR(255) NOT NULL,
    tipo_accidente VARCHAR(100) NOT NULL,
    severidad ENUM('Leve', 'Moderado', 'Grave', 'Fatal') NOT NULL,
    vehiculos_implicados INT DEFAULT 1,
    heridos INT DEFAULT 0,
    fallecidos INT DEFAULT 0,
    condiciones_climaticas VARCHAR(100),
    descripcion TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Tabla de vehículos implicados
CREATE TABLE IF NOT EXISTS vehiculos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    accidente_id INT NOT NULL,
    tipo_vehiculo VARCHAR(50) NOT NULL,
    marca VARCHAR(50),
    modelo VARCHAR(50),
    año INT,
    seguro VARCHAR(100),
    FOREIGN KEY (accidente_id) REFERENCES accidentes(id) ON DELETE CASCADE
);

-- Tabla de personas implicadas
CREATE TABLE IF NOT EXISTS personas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    accidente_id INT NOT NULL,
    vehiculo_id INT,
    nombre VARCHAR(100) NOT NULL,
    edad INT,
    genero ENUM('M', 'F', 'Otro'),
    tipo ENUM('Conductor', 'Pasajero', 'Peatón'),
    lesion ENUM('Ninguna', 'Leve', 'Grave', 'Fatal'),
    alcohol BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (accidente_id) REFERENCES accidentes(id) ON DELETE CASCADE,
    FOREIGN KEY (vehiculo_id) REFERENCES vehiculos(id) ON DELETE SET NULL
);

-- Insertar datos de ejemplo
INSERT INTO accidentes (fecha, hora, ubicacion, tipo_accidente, severidad, vehiculos_implicados, heridos, fallecidos, condiciones_climaticas) VALUES
('2024-01-15', '08:30:00', 'Av. Principal esq. Calle Secundaria', 'Colisión frontal', 'Moderado', 2, 3, 0, 'Lluvia ligera'),
('2024-01-16', '14:45:00', 'Carrera 10 #20-30', 'Choque lateral', 'Leve', 2, 1, 0, 'Soleado'),
('2024-01-17', '22:15:00', 'Autopista Norte Km 25', 'Volcamiento', 'Grave', 1, 2, 1, 'Noche cerrada');

-- Crear índices para mejorar el rendimiento
CREATE INDEX idx_accidentes_fecha ON accidentes(fecha);
CREATE INDEX idx_accidentes_ubicacion ON accidentes(ubicacion);
CREATE INDEX idx_accidentes_severidad ON accidentes(severidad);
CREATE INDEX idx_personas_accidente ON personas(accidente_id);
CREATE INDEX idx_vehiculos_accidente ON vehiculos(accidente_id);