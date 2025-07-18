# src/gui.py
import tkinter as tk
from tkinter import messagebox
import requests

def analizar_sentimiento():
    texto = entrada.get()
    if not texto.strip():
        messagebox.showwarning("Campo vacío", "Por favor ingresa un texto.")
        return

    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"text": texto}
        )
        response.raise_for_status()
        pred = response.json()['prediction']
        if pred == 1:
            resultado['text'] = "✅ Positivo 😊"
            resultado['fg'] = 'green'
        else:
            resultado['text'] = "❌ Negativo 😞"
            resultado['fg'] = 'red'
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error de conexión", f"No se pudo conectar con el API.\n{e}")

# Crear ventana
root = tk.Tk()
root.title("Analizador de Sentimientos")
root.geometry("400x200")
root.resizable(False, False)

# Widgets
titulo = tk.Label(root, text="Escribe tu reseña:", font=("Helvetica", 14))
titulo.pack(pady=10)

entrada = tk.Entry(root, width=50)
entrada.pack(padx=10, pady=5)

boton = tk.Button(root, text="Analizar", command=analizar_sentimiento, bg="#4CAF50", fg="white", width=15)
boton.pack(pady=10)

resultado = tk.Label(root, text="", font=("Helvetica", 16))
resultado.pack(pady=10)

# Iniciar loop
root.mainloop()
