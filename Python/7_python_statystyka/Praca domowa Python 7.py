#+++++++++++++++PYTHON++++++++++++++++++++++

# Polecenie nr 1

def wiersze(tekst, liczba):
    if type(tekst) != type(str(tekst)):
        y = str(tekst)
    else:
        y = tekst
    if type(liczba) == type(int(liczba)):
        for i in range(liczba):
            print(y)
    else:
        print("Liczba musi mieć typ int!")

#wiersze(input("podaj cokolwiek: \n"), 5)

#Polecenie nr. 2

def zaokraglenie(liczba, zaokr):
   if len(str(liczba+2)) > zaokr:
        x = round(liczba, zaokr)
        return "Liczba "+str(liczba)+" została zaokrąglona do "+str(x)+" z precyzją "+str(zaokr)
   else:
       return "Za duza precyzja"

#print(zaokraglenie(1.34245, 50))

#Polecenie 3

def Monty(string):
    if type(string) != type(str(string)):
        x = str(string)
    else:
        x = string
    if "Monty" in x:
        return "Monty znajduje sie w zdaniu!"
    else:
        return "Brak słowa Monty w zdaniu"

#print(Monty("Monty to mistrz"))

#Polecenie nr 4:

def ASCI (cos):
    if type(cos) != type(str(cos)):
        x = str(cos)
    else:
        x = cos
    lista_x = []
    for i in x:
        if type(i) != type(str(i)):
            string_i = str(i)
        else:
            string_i = i
        lista_x.append(string_i)
    #print(lista_x)
    string_ASCI = ""
    for i in lista_x:
        string_ASCI += str(ord(i))
    return string_ASCI

#print(ASCI("hjdha a ad ad  asd"))


#++++++++++++++++STATYSTYKA+++++++++++++++++
import pandas as pd
#import numpy as np
data = pd.read_csv(r'C:\Users\Mateusz\Desktop\user-languages.csv', delimiter=',' , dtype={'visual-studio':float, 'c#':float})
data = data[['user_id','visual-studio', 'c#']]
data.rename(columns={'c#':'cc',
                     'visual-studio':'vs'}, inplace=True)

print(data)
x = data.cc.count()
x_float = float(x)
y_float = float(data.vs.count())
all_float = float(data.user_id.count())
data['cc_count'] = pd.Series(float(0), index=data.index, dtype =float)
data['vs_count'] = pd.Series(float(0), index=data.index, dtype =float)
data['cc_and_vs']= pd.Series(float(0), index=data.index, dtype =float)

for ind in data.index:
    if data['cc'][ind]>0:
        data['cc_count'][ind] = data['cc_count'][ind]+1
    elif data['vs'][ind]>0:
        data['vs_count'][ind] = data['vs_count'][ind]+1
    elif data['cc'][ind]>0 and data['vs'][ind]>0:
        data['cc_and_vs'][ind] = data['cc_and_vs'][ind]+1
        #print(data2['cc_count'][ind], data2['cc'][ind], data2['vs'][ind])

vs_co = data.vs_count.sum()
cc_co = data.cc_count.sum()
vac_co = data.cc_and_vs.sum()

print("vs_co: ", vs_co, "\ncc_co", cc_co, "\nvac_co: ", vac_co, "\nall: ", all_float)
#print(data)

# prawdopodobienstwo nie uzwyania VS
p1 = (all_float-vs_co)/all_float
print(p1)
# prawdopodobienstwo nieuzwyania C#
p2 = (all_float-cc_co)/all_float
print(p2)
#prawdopodobienstwo uzywnia c# przy nieuzywaniu vs
p3 = cc_co/(all_float-vac_co)
print(p3)
##prawdopodobienstwo uzywnia vs przy nieuzywaniu c#
p4 = vs_co/(all_float-vac_co)
print(p4)