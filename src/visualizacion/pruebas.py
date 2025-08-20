from modelos.modelo_accidentes import ModeloAccidentes

# Inicializar
modelo = ModeloAccidentes()
df = modelo.cargar_datos()
X, y = modelo.preparar_datos(df)

# Entrenar
modelo.entrenar_modelo(X, y)

# Probar predicción: martes (1), 18:00, San José, 5mm lluvia
prob = modelo.predecir(hora=18, dia_semana=1, provincia="San José", precip_acum=5.0)
print(f"🔮 Probabilidad de accidente: {prob:.2%}")
