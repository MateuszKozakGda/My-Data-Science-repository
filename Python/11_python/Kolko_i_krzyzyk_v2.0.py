import random


class rzut_moneta(object):
    def __init__(self):
        print("Rzut monetą. \n")
        self.wybor = (0, 1)

    def rzut(self):
        self.rzut = random.choice(self.wybor)
        # print("Następuje rzut i... wyleciał: ")
        return self.rzut


class tictactoe(object):
    def __init__(self):
        print("Gra w kółko i krzyżyk!")
        self.ilosc_pol = 9
        self.puste_pole = " "
        self.text_file = open("tictactoe_wyniki.txt", "a+")

    def instrukcja(self):
        print(
            """Witaj w grze kółko i krzyżyk! Będziesz miał okazje zagrać z komputerem w tę super grę! 
            Gwarantuję ci że nie uda Ci się wygrać! No chyba że się mylę... ale ja nigdy się nie mylę!
    
            Swoje posunięcie wskażesz poprzez wrpowadzenie liczby z zakresu 0-8.
            Liczba ta odpowiada pozycji na planszy zgodnie z poniższym schematem:
    
                        0 | 1 | 2
                        ---------
                        3 | 4 | 5
                        ---------
                        6 | 7 | 8
    
            Przygotuj się na straszne baty! \n
            """)

    def kto_zaczyna(self, pytanie):
        self.respone = None
        while self.respone not in ("t", "n"):
            self.respone = input(pytanie).lower()
        return self.respone

    def ktore_pole(self, pytanie, zero, ilosc_wolnych_pol):
        self.podane_pole = None
        while self.podane_pole not in range(zero, ilosc_wolnych_pol):
            self.podane_pole = int(input(pytanie))
        return self.podane_pole

    def nowa_plansza(self):
        self.plansza = []
        for pole in range(self.ilosc_pol):
            self.plansza.append(self.puste_pole)
        return self.plansza

    def wyswietl_plansze(self, plansza):
        print("Oto jak wygląda teraz plansza, zastanów się nad następnym ruchem!! \n\n")
        print("\n\t", plansza[0], "|", plansza[1], "|", plansza[2])
        print("\t-----------")
        print("\t", plansza[3], "|", plansza[4], "|", plansza[5])
        print("\t-----------")
        print("\t", plansza[6], "|", plansza[7], "|", plansza[8], "\n")

    def kto_kolko_a_kto_krzyzyk(self):
        self.zaczyna = self.kto_zaczyna("Czy chcesz mieć prawo pierwszego ruchu? (t/n): \n")
        if self.zaczyna == "t":
            print("Pierwszy ruch należy do Ciebie, powodzenia!")
            self.czlowiek = "X"
            self.computer = "O"
        elif self.zaczyna == "n":
            print("A więc to ja zaczynam! Nie masz ze mną szans...")
            self.czlowiek = "O"
            self.computer = "X"
        return self.czlowiek, self.computer

    def dozwolony_ruch(self, plansza):
        self.ruch2 = []
        for self.pole in range(self.ilosc_pol):
            if plansza[self.pole] == self.puste_pole:
                self.ruch2.append(self.pole)
        return self.ruch2

    def jak_wygrac(self, plansza):
        self.ruchy_wygrane = ((0, 1, 2),
                              (3, 4, 5),
                              (6, 7, 8),
                              (0, 3, 6),
                              (1, 4, 7),
                              (2, 5, 8),
                              (0, 4, 8),
                              (2, 4, 6))
        for self.row in self.ruchy_wygrane:
            if plansza[self.row[0]] == plansza[self.row[1]] == plansza[self.row[2]] != self.puste_pole:
                self.winner = plansza[self.row[0]]
                return self.winner
            if self.puste_pole not in plansza:
                return "remis"
        return None

    def ruch_czlowieka(self, plansza):
        self.dozwolone = self.dozwolony_ruch(plansza)
        print(f"Dozwolone pola to: \n {self.dozwolone}")
        self.ruchy = None
        while self.ruchy not in self.dozwolone:
            self.ruchy = self.ktore_pole("Na które pole chcesz postawić swój znak?\n", 0, self.ilosc_pol)
            print(f"Chcesz wykonać ruch na pole {self.ruchy}? Dobrze się składa bo jest wolne!")
            if self.ruchy not in self.dozwolone:
                print("\nTo pole jest już zajęte niemądry człowieku. Wybierz inne. \n")
        print("Znakomicie....")
        return self.ruchy

    def ruch_komputera(self, plansza, komputer, czlowiek):
        """utwórz kopię roboczą, ponieważ funkcja będzie zmieniać listę.  !!!!!!
        # kopiuje cała zawartosc listy board. Jeżeli wartosc jest mutowalna i wykorzystywana
        w innej funkcji zawsze lepiej jest uzywac kopii niz wartosci wyjsciowej!!!"""

        plansza = plansza[:]

        # najlepsze pozycje wg mnie :) - komputer bedzie wybieral w kolejnosci najlepsze pozycje w ktorych zrobi ruch
        gdy_komputer_X = [(4, 2, 5), (4, 6, 8), (4, 8, 2), (4, 6, 8), (4, 6, 0), (4, 0, 6), (4, 6, 8), (4, 8, 6),
                          (4, 8, 2), (4, 2, 8), (4, 0, 6), (4, 6, 0), (7,3,5), (1,3,7)]
        gdy_komputer_O = (0, 2, 6, 8)

        # Sprawdzam cy komputer moze wygrac
        print("wybieram pole numer")

        for dobry_ruch in self.dozwolony_ruch(plansza):
            plansza[dobry_ruch] = komputer
            if self.jak_wygrac(plansza) == komputer:
                print(dobry_ruch)
                return dobry_ruch
            ##wycofanie tego ruchy aby zresetować plansze
            plansza[dobry_ruch] = self.puste_pole
        # sprawdzamy czy w następnym ruchu może wygrać gracz. Komputer musi koniecznie zablokować ten ruch.
        # Kod w pętli dobiera kolejne elementy listy prawidłowych ruchów umieszczajac ruch czlowieka w kazdym po koleji pustym polu i sprawdza jego mozliwosc zwyciestwa
        # Jezeli sie okarze ruch da czlowiekowi zwyciestwo, komputer wykona ten ruch jako pierwszy i zablokuje pole na ruch czlowieka.
        # Jezeli nie bedzie takiej mozlwiosci, komputer wycofa ruch i sprawdzi kolejny najlepszy ruch z listy czy jest mozliwy do wykonania

        for dobry_ruch2 in self.dozwolony_ruch(plansza):
            plansza[dobry_ruch2] = czlowiek
            if self.jak_wygrac(plansza) == czlowiek:
                print(dobry_ruch2)
                return dobry_ruch2
            # ruch został sprawdzony
            plansza[dobry_ruch2] = self.puste_pole

        ##jezeli nikt nie moze wygrac to komputer bedzie sprawdzal gdzie mozne postawic kolejny ruch o ile sa jeszcze wolne pola

        if komputer == "X":
            for i in range(len(gdy_komputer_X)):
                for ruch_x1 in gdy_komputer_X[i]:
                    if ruch_x1 in self.dozwolony_ruch(plansza):
                        print(ruch_x1)
                        return ruch_x1
        elif komputer == "O":
            for self.ruchy_dozwolone in self.dozwolony_ruch(plansza):
                print(f"tratata :{self.ruchy_dozwolone}")
                if self.ruchy_dozwolone in gdy_komputer_O:
                    for ruch_x2 in gdy_komputer_O:
                        if ruch_x2 in self.dozwolony_ruch(plansza):
                            print(ruch_x2)
                            return ruch_x2
                else:
                    print("robie ruch na pałe!")
                    self.pozostale = random.choice(self.dozwolony_ruch(plansza))
                    return self.pozostale

    def koljeka(self, kolejka):
        """Zmien wykonawce ruchu"""
        if kolejka == "X":
            return "O"
        else:
            return "X"

    def gratulacje(self, zwyciesca, komputer, czlowiek):
        """Pogratuluj zwyciezcy"""
        if zwyciesca != "remis":
            print(zwyciesca, " jest zwyciezcą!!\n")
        else:
            print("REMIS")

        if zwyciesca == komputer:
            print("Jak podejżewałem, nie miałes ze mna żadnych szans!!")

        elif zwyciesca == czlowiek:
            print(
                "No nie! To niemożliwe! Jak mogłes mnie pokonać!!")

        elif zwyciesca == "remis":
            print("Jest tak jak mówiłem, na więcej nie mogło Cię być stać. Remis to i tak za dużo jak na Ciebie!!")

    def gra(self, moneta):
        self.instrukcja()
        self.moneta = None
        while not self.moneta in (0,1):
            self.moneta = int(input("Wybierz orzeł(1) czy reszka(0)?\n"))
        if moneta == 1:
            self.co_wylecialo = "Orła"
        else:
            self.co_wylecialo = "Reszkę"
        print(f"Rzut wylosował: {self.co_wylecialo}")

        if self.moneta == moneta:
            self.czlowiek = "X"
            self.komputer = "O"
        else:
            self.czlowiek = "O"
            self.komputer = "X"
        # self.czlowiek, self.komputer = self.__kto_kolko_a_kto_krzyzyk()
        self.kolejka = "X"
        self.plansza = self.nowa_plansza()
        self.wyswietl_plansze(self.plansza)
        print(f"Czlowiek: {self.czlowiek}, komputer: {self.komputer}")
        print(f"test: {self.jak_wygrac(self.plansza)}")

        while not self.jak_wygrac(self.plansza):
            if self.kolejka == self.czlowiek:
                self.ruch3 = self.ruch_czlowieka(self.plansza)
                self.plansza[self.ruch3] = self.czlowiek
            else:
                self.ruch4 = self.ruch_komputera(self.plansza, self.komputer, self.czlowiek)
                self.plansza[self.ruch4] = self.komputer
            self.wyswietl_plansze(self.plansza)
            self.kolejka = self.koljeka(self.kolejka)

        self.win = self.jak_wygrac(self.plansza)
        self.gratulacje(self.win, self.komputer, self.czlowiek)
        # self.zapisywanie = self.win
        self.text_file.writelines(self.win)
        self.text_file.close()


class Tic_tac_Komputery(tictactoe):

    def ruch_komputera2(self, plansza, komputer1, komputer2):
        """utwórz kopię roboczą, ponieważ funkcja będzie zmieniać listę.  !!!!!!
        # kopiuje cała zawartosc listy board. Jeżeli wartosc jest mutowalna i wykorzystywana
        w innej funkcji zawsze lepiej jest uzywac kopii niz wartosci wyjsciowej!!!"""

        plansza = plansza[:]

        # najlepsze pozycje wg mnie :) - komputer bedzie wybieral w kolejnosci najlepsze pozycje w ktorych zrobi ruch
        gdy_komputer_X = [(4, 2, 5), (4, 6, 8), (4, 8, 2), (4, 6, 8), (4, 6, 0), (4, 0, 6), (4, 6, 8), (4, 8, 6),
                          (4, 8, 2), (4, 2, 8), (4, 0, 6), (4, 6, 0), (1,3,7), (7,3,5), (7,5,1)]
        gdy_komputer_O = (0, 2, 6, 8)

        # Sprawdzam cy komputer moze wygrac
        print("wybieram pole numer")

        for dobry_ruch in self.dozwolony_ruch(plansza):
            plansza[dobry_ruch] = komputer1
            if self.jak_wygrac(plansza) == komputer1:
                print(dobry_ruch)
                return dobry_ruch
            ##wycofanie tego ruchy aby zresetować plansze
            plansza[dobry_ruch] = self.puste_pole
        # sprawdzamy czy w następnym ruchu może wygrać gracz. Komputer musi koniecznie zablokować ten ruch.
        # Kod w pętli dobiera kolejne elementy listy prawidłowych ruchów umieszczajac ruch czlowieka w kazdym po koleji pustym polu i sprawdza jego mozliwosc zwyciestwa
        # Jezeli sie okarze ruch da czlowiekowi zwyciestwo, komputer wykona ten ruch jako pierwszy i zablokuje pole na ruch czlowieka.
        # Jezeli nie bedzie takiej mozlwiosci, komputer wycofa ruch i sprawdzi kolejny najlepszy ruch z listy czy jest mozliwy do wykonania

        for dobry_ruch2 in self.dozwolony_ruch(plansza):
            plansza[dobry_ruch2] = komputer2
            if self.jak_wygrac(plansza) == komputer2:
                print(dobry_ruch2)
                return dobry_ruch2
            # ruch został sprawdzony
            plansza[dobry_ruch2] = self.puste_pole

        ##jezeli nikt nie moze wygrac to komputer bedzie sprawdzal gdzie mozne postawic kolejny ruch o ile sa jeszcze wolne pola

        if komputer2 == "X":
            for i in range(len(gdy_komputer_X)):
                for ruch_x1 in gdy_komputer_X[i]:
                    if ruch_x1 in self.dozwolony_ruch(plansza):
                        print(ruch_x1)
                        return ruch_x1
        elif komputer2 == "O":
            for self.ruchy_dozwolone in self.dozwolony_ruch(plansza):
                print(f"srututu :{self.ruchy_dozwolone}")
                if self.ruchy_dozwolone in gdy_komputer_O:
                    for ruch_x2 in gdy_komputer_O:
                        if ruch_x2 in self.dozwolony_ruch(plansza):
                            print(ruch_x2)
                            return ruch_x2
                else:
                    print("robie ruch na pałe!")
                    self.pozostale = random.choice(self.dozwolony_ruch(plansza))
                    print(f"Pozostałe ruchy: {self.pozostale}")
                    return self.pozostale

    def gratulacje_komputerow(self, zwyciesca, komputer1, komputer2):

        """Pogratuluj zwyciezcy"""
        if zwyciesca != "remis":
            print(zwyciesca, " jest zwyciezcą!!\n")
        else:
            print("REMIS")

        if zwyciesca == komputer1:
            print("komputer 1 zwyciężył!!")

        elif zwyciesca == komputer2:
            print("Komputer 2 zwyciężył!!")

        elif zwyciesca == "remis":
            print("Trafiło na równego z równym!")

    def gra_komputerow(self, moneta):
        self.instrukcja()
        self.choice = (0, 1)
        self.wybor_komputera_1 = random.choice(self.choice)
        self.o = "Orzeł"
        self.r = "Reszka"
        print(f"Komputer 1 wybrał: {self.o if self.wybor_komputera_1 == 1 else self.r}")

        if moneta == 1:
            self.co_wylecialo = "Orła"
        else:
            self.co_wylecialo = "Reszkę"
        print(f"Rzut wylosował: {self.co_wylecialo}")

        if self.wybor_komputera_1 == moneta:
            self.komputer1 = "X"
            self.komputer2 = "O"
        else:
            self.komputer1 = "O"
            self.komputer2 = "X"
        self.kolejka = "X"
        self.plansza = self.nowa_plansza()
        print(f"Komputer 1: {self.komputer1}, komputer2: {self.komputer2}")

        while not self.jak_wygrac(self.plansza):
            if self.kolejka == self.komputer1:
                self.ruch5 = self.ruch_komputera(self.plansza, self.komputer1, self.komputer2)
                self.plansza[self.ruch5] = self.komputer1
            else:
                self.ruch4 = self.ruch_komputera2(self.plansza, self.komputer1, self.komputer2)
                self.plansza[self.ruch4] = self.komputer2
            self.wyswietl_plansze(self.plansza)
            self.kolejka = self.koljeka(self.kolejka)

        self.win = self.jak_wygrac(self.plansza)
        self.gratulacje_komputerow(self.win, self.komputer1, self.komputer2)
        # self.zapisywanie = self.win
        self.text_file.writelines(self.win)
        self.text_file.close()

class gra_totalna(Tic_tac_Komputery):

    def czlowiek_czy_komputery(self):
        self.wybieracz = None
        print("""Wybór graczy:
                       1 - Gra gracz vs komputer
                       2 - Komputer vs Komputer
               """)
        while not self.wybieracz in (1, 2):
            self.wybieracz = int(input("Wybierz odpowiednią opcję: \n"))

        if self.wybieracz == 1:
            return True
        else:
            return False

    def gramy(self, odpowiedz):
        self.rzut_moneta_gra = rzut_moneta().rzut()
        if odpowiedz:
            self.gra(self.rzut_moneta_gra)
        else:
            self.gra_komputerow(self.rzut_moneta_gra)


z = gra_totalna()
pyt = z.czlowiek_czy_komputery()
z.gramy(pyt)
