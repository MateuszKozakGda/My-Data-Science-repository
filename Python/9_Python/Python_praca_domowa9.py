
### Zadanie 1
class jakas_klasa(object):
    def __init__(self, wypisz_imie):
        print("Jestem sobie jakąs klasą!!")
        self.wypisz_imie = wypisz_imie
        print(f"Wyswietlam imie: {wypisz_imie}")

    def atrybut(self, napis):
        print(f"Wyswitlam jakis napis, ten napis to {napis}")
        x = input("Podaj cos co mam zwrócić!\n")
        return x

z = jakas_klasa("Jarek")
z2 = jakas_klasa("Marek")

z.atrybut("super klasa")
z2.atrybut("ta jest fajniejsza")

### zadanie 2
class jakas_klasa_dziecko(jakas_klasa):
    def atrybut(self, napis, liczba):
        print((f"Wyswitlam jakis napis, ten napis to: {napis}"))
        print(f"Dodatkowo zwrócę liczbę {liczba}")
        return napis, float(liczba)

z3 = jakas_klasa_dziecko("Darek")
z4 = jakas_klasa_dziecko("Mateusz")

z3.atrybut("ekstra", 2)
z4.atrybut("super ekstra", 4)

### Zadanie 3
class jakas_klasa2(object):
    def __init__(self, wypisz_imie):
        print("Jestem sobie jakąs klasą!!")
        self.wypisz_imie = wypisz_imie
        self.lista = []
        print(f"Wyswietlam imie: {wypisz_imie}")

    def atrybut(self, napis):
        print(f"Wyswitlam jakis napis, ten napis to {napis}")
        x = input("Podaj cos co mam zwrócić!\n")
        self.napis = napis
        return self.napis

    def zmien_napis(self):
        print("Zmienię wypisane imię, a stare dorzucę do listy będącej historią")
        self.lista.append(self.wypisz_imie)
        self.wypisz_imie = input("Podaj nowe imię: \n")
        print(f"Historia starych imion: \n{self.lista}")
        print(f"Bierzące imię to: \n{self.wypisz_imie}")

z = jakas_klasa2("jarek")
z.zmien_napis()


for i in range(0,4):
    z.zmien_napis()

### Zadanie 4
z.atrybut("super iterator")
i = 0
for atributes, its_value in vars(z).items():
    print(f"Pętla nr: {i}")
    i+=1
    print(f"Atrybut {atributes} ma wartosć: {its_value}")