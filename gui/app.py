
import tkinter as tk
from tkinter import ttk, messagebox
from src.strategy import BertStrategy, RegressionStrategy, LSTMStrategy

def predict():
    task = task_var.get()
    model_type = model_var.get()
    text = text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Erreur", "Le texte est vide!")
        return
    if model_type == "bert":
        strategy = BertStrategy()
    elif model_type == "regression":
        strategy = RegressionStrategy(task)
    elif model_type == "lstm":
        strategy = LSTMStrategy(task)
    else:
        messagebox.showerror("Erreur", "Modèle inconnu")
        return
    prediction = strategy.classify(text)
    result_label.config(text=f"Prédiction: {prediction}")

app = tk.Tk()
app.title("Classification Textuelle")

# Sélection de la tâche
tk.Label(app, text="Tâche (darija, sentiment, spam, toxic):").pack()
task_var = tk.StringVar(value="darija")
task_entry = ttk.Entry(app, textvariable=task_var)
task_entry.pack()

# Sélection du modèle
tk.Label(app, text="Modèle (bert, regression, lstm):").pack()
model_var = tk.StringVar(value="bert")
model_entry = ttk.Entry(app, textvariable=model_var)
model_entry.pack()

# Zone de texte
tk.Label(app, text="Texte:").pack()
text_entry = tk.Text(app, height=10, width=50)
text_entry.pack()

# Bouton de prédiction
predict_button = ttk.Button(app, text="Prédire", command=predict)
predict_button.pack()

# Label de résultat
result_label = tk.Label(app, text="Prédiction:")
result_label.pack()

app.mainloop()
