-- Consultas comunes para el análisis de accidentes

-- 1. Accidentes por severidad
SELECT severidad, COUNT(*) as total
FROM accidentes
GROUP BY severidad
ORDER BY total DESC;

-- 2. Accidentes por mes
SELECT MONTH(fecha) as mes, COUNT(*) as total_accidentes
FROM accidentes
GROUP BY MONTH(fecha)
ORDER BY mes;

-- 3. Accidentes por condiciones climáticas
SELECT condiciones_climaticas, COUNT(*) as total
FROM accidentes
WHERE condiciones_climaticas IS NOT NULL
GROUP BY condiciones_climaticas
ORDER BY total DESC;

-- 4. Top 5 ubicaciones con más accidentes
SELECT ubicacion, COUNT(*) as total_accidentes
FROM accidentes
GROUP BY ubicacion
ORDER BY total_accidentes DESC
LIMIT 5;

-- 5. Estadísticas de víctimas
SELECT
    SUM(heridos) as total_heridos,
    SUM(fallecidos) as total_fallecidos,
    AVG(heridos) as promedio_heridos_por_accidente
FROM accidentes;

-- 6. Accidentes por hora del día
SELECT
    HOUR(hora) as hora_del_dia,
    COUNT(*) as total_accidentes
FROM accidentes
GROUP BY HOUR(hora)
ORDER BY hora_del_dia;