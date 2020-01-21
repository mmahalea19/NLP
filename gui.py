from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *


def clicked():
    # messagebox.showinfo('Result', 'SPAM')
    message=txtInput.get("1.0",END);
    filter=strategyCombo.get();
    strategy="pca" * chk_state1.get() +"lda" * chk_state2.get()
    nrFeats=int(nrFeatIn.get())
    print("d")


window = Tk()
window.title('NLP Project')
window.geometry('600x800')

titleLbl = Label(window, text="Spam or Ham?")
titleLbl.place(x=250, y=10, width=100, height=30)

instrLbl = Label(window, text="Input email:")
instrLbl.place(x=250, y=90, width=100, height=30)

txtInput = Text(window, width=50, height=20)
txtInput.place(x=100, y=130, width=400, height=270)

strategyLbl = Label(window, text="Choose Strategy:")
strategyLbl.place(x=100, y=410, width=150, height=30)

txtInput = Text(window, width=50, height=20)
txtInput.place(x=100, y=130, width=400, height=270)

strategyCombo = Combobox(window)
strategyCombo['values'] = ('Naive Bayes', 'Liner SVC',"Decision Tree","Random Forest",)
strategyCombo.place(x=200, y=410, width=150, height=30)
strategyCombo.current(0)
chk_state1 = BooleanVar()

chk_state1.set(False)  # set check state

chk_pca = Checkbutton(window, text='PCA', var=chk_state1)

chk_pca.place(x=100, y=450, width=150, height=30)

chk_state2 = BooleanVar()

chk_state2.set(False)  # set check state

chk_lda = Checkbutton(window, text='LDA', var=chk_state2)

chk_lda.place(x=100, y=490, width=150, height=30)


nrfeatLabel=Label(window, text='Nr. features:')
nrfeatLabel.place(x=100, y=530, width=80, height=30)
nrFeatIn = Combobox(window)
nrFeatIn['values'] = (3000,2000,1000,500)
nrFeatIn.place(x=180, y=530, width=150, height=30)
nrFeatIn.current(0)


nrfeatLabel=Label(window, text='Removal strategies:')
nrfeatLabel.place(x=100, y=570, width=150, height=30)



chk_state3 = BooleanVar()

chk_state3.set(False)  # set check state

chk_stopword = Checkbutton(window, text='stopword', var=chk_state3)
chk_stopword.place(x=100, y=590, width=80, height=30)
chk_state4 = BooleanVar()

chk_state4.set(False)  # set check state

chk_link = Checkbutton(window, text='link', var=chk_state4)
chk_link.place(x=100, y=620, width=80, height=30)
chk_state5 = BooleanVar()

chk_state5.set(False)  # set check state

chk_symbol = Checkbutton(window, text='symbol', var=chk_state5)
chk_symbol.place(x=100, y=650, width=80, height=30)










btn = Button(window, text="Classify", command=clicked)
btn.place(x=100, y=740, width=80, height=30)


window.mainloop()
