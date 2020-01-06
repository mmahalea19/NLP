from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *


def clicked():
    messagebox.showinfo('Result', 'SPAM')


window = Tk()
window.title('NLP Project')
window.geometry('600x600')

titleLbl = Label(window, text="Spam or Ham?")
titleLbl.place(x=250, y=10, width=100, height=30)

instrLbl = Label(window, text="Input email:")
instrLbl.place(x=250, y=90, width=100, height=30)

txtInput = Text(window, width=50, height=20)
txtInput.place(x=100, y=130, width=400, height=270)

strategyLbl = Label(window, text="Choose Strategy:")
strategyLbl.place(x=100, y=410, width=150, height=30)

strategyCombo = Combobox(window)
strategyCombo['values'] = ('Naive Bayes', 'Random Forest')
strategyCombo.place(x=200, y=410, width=150, height=30)

btn = Button(window, text="Classify", command=clicked)
btn.place(x=100, y=500, width=80, height=30)

chkLDA = Checkbutton(window, text='Use PCA')
chkLDA.place(x=100, y=450, width=150, height=30)

window.mainloop()
