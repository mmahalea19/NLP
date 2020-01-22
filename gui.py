from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
import pickle
from all import *
def run_algorithm(message,featureExtractor,classifierName,feature_reduction,nrFeats,filters):
    filename = "./Out/Bundle_{}_{}_{}_{}".format("_".join(filters), featureExtractor,nrFeats, feature_reduction);
    with open(filename, 'rb') as fi:
        [classifiers,classifierLabels,vectorizer,decomposer] = pickle.load(fi)
    test_matrix=[]
    message=removeWords([message],filters);#filter the message first
    if featureExtractor == 'IDF':#apply the correct feature extractor
        test_matrix=vectorizer.transform(message).todense().getA()
    elif featureExtractor =="FFeats":
        compute_enhanced_vector(message,test_matrix)

    if nrFeats is not "":
        test_matrix=reduceFeatures(test_matrix,nrFeats)
    if feature_reduction is not "":
        test_matrix=decomposer.transform(test_matrix)

    classifierIndex=classifierLabels.index(classifierName);
    result=classifiers[classifierIndex].predict(test_matrix);



    return result


def gather_input():
    # messagebox.showinfo('Result', 'SPAM')
    message=txtInput.get("1.0",END);
    classifierName=strategyCombo.get();
    feature_reduction=""+"pca" * chk_state1.get() +"lda" * chk_state2.get()
    nrFeats=int(nrFeatIn.get()) if nrFeatIn.get() is not "" else ""
    filters=[];
    featureExtractor=featureExtractionBox.get();

    if(chk_state3.get()):
        filters.append("stopword")
    if (chk_state4.get()):
        filters.append("link")
    if (chk_state5.get()):
        filters.append("symbol")
    result=run_algorithm(message,featureExtractor,classifierName,feature_reduction,nrFeats,filters)
    messagebox.showinfo('Result', 'HAM') if result==0 else messagebox.showinfo('Result', 'SPAM');


window = Tk()
window.title('NLP Project')
window.geometry('600x900')

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
strategyCombo['values'] = ("Gaussian", "LinearSVC", "Decision Tree", "Random Forest")
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
nrFeatIn['values'] = ("",3000,2000,1000,500)
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




featureExtractionLabel = Label(window, text="Choose Feature extraction Strategy:")
featureExtractionLabel.place(x=100, y=700, width=260, height=30)
featureExtractionBox = Combobox(window)
featureExtractionBox['values'] = ("IDF","FFeats")
featureExtractionBox.place(x=360, y=700, width=150, height=30)
featureExtractionBox.current(0)








btn = Button(window, text="Classify", command=gather_input)
btn.place(x=100, y=780, width=80, height=30)


window.mainloop()
