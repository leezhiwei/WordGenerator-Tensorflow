import tkinter as tk
window = tk.Tk()
text = "Hello World"
greeting= tk.Label(text=text,
		width=10,
		height=3)
button = tk.Button(text="Generate a new word!",
		width=22,
		height=3,
		bg="black",
		fg="white")
greeting.pack()
button.pack()
window.mainloop()
