class tictactoe(object):
    def __init__(self):
        print("Gra w kółko i krzyżyk!")
        self.__ilosc_pol = 9
        self.__puste_pole = " "
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

    def __kto_zaczyna(self, pytanie):
        self.respone = None
        while self.respone not in ("t", "n"):
            self.respone = input(pytanie).lower()
        return self.respone

    def __ktore_pole(self, pytanie ,zero, ilosc_wolnych_pol):
        self.podane_pole = None
        while self.podane_pole not in range(zero, ilosc_wolnych_pol):
            self.podane_pole = int(input(pytanie))
        return self.podane_pole

    def __nowa_plansza(self):
        self.plansza = []
        for pole in range(self.__ilosc_pol):
            self.plansza.append(self.__puste_pole)
        return self.plansza

    def __wyswietl_plansze(self, plansza):
        print("Oto jak wygląda teraz plansza, zastanów się nad następnym ruchem!! \n\n")
        print("\n\t", plansza[0], "|", plansza[1], "|", plansza[2])
        print("\t-----------")
        print("\t", plansza[3], "|", plansza[4], "|", plansza[5])
        print("\t-----------")
        print("\t", plansza[6], "|", plansza[7], "|", plansza[8], "\n")

    def __kto_kolko_a_kto_krzyzyk(self):
        self.zaczyna = self.__kto_zaczyna("Czy chcesz mieć prawo pierwszego ruchu? (t/n): \n")
        if self.zaczyna == "t":
            print("Pierwszy ruch należy do Ciebie, powodzenia!")
            self.czlowiek = "X"
            self.computer = "O"
        elif self.zaczyna == "n":
            print("A więc to ja zaczynam! Nie masz ze mną szans...")
            self.czlowiek = "O"
            self.computer = "X"
        return self.czlowiek, self.computer

    def __dozwolony_ruch(self,plansza):
        self.ruch2 = []
        for self.pole in range(self.__ilosc_pol):
            if plansza[self.pole] == self.__puste_pole:
                self.ruch2.append(self.pole)
        return self.ruch2

    def __jak_wygrac(self, plansza):
        self.ruchy_wygrane = ((0, 1, 2),
                (3, 4 ,5),
                (6, 7, 8),
                (0, 3, 6),
                (1, 4, 7),
                (2, 5, 8),
                (0, 4, 8),
                (2, 4, 6))
        for self.row in self.ruchy_wygrane:
            if plansza[self.row[0]] == plansza[self.row[1]] == plansza[self.row[2]] != self.__puste_pole:
                self.winner = plansza[self.row[0]]
                return self.winner
            if self.__puste_pole not in plansza:
                return "remis"
        return None

    def __ruch_czlowieka(self, plansza):
        self.dozwolone = self.__dozwolony_ruch(plansza)
        print(f"Dozwolone pola to: \n {self.dozwolone}")
        self.ruchy = None
        while self.ruchy not in self.dozwolone:
            self.ruchy = self.__ktore_pole("Na które pole chcesz postawić swój znak?\n", 0, self.__ilosc_pol)
            print(f"Chcesz wykonać ruch na pole {self.ruchy}? Dobrze się składa bo jest wolne!")
            if self.ruchy not in self.dozwolone:
                print("\nTo pole jest już zajęte niemądry człowieku. Wybierz inne. \n")
        print("Znakomicie....")
        return self.ruchy

    def __ruch_komputera(self, plansza, komputer, czlowiek):
        """utwórz kopię roboczą, ponieważ funkcja będzie zmieniać listę.  !!!!!!
        # kopiuje cała zawartosc listy board. Jeżeli wartosc jest mutowalna i wykorzystywana
        w innej funkcji zawsze lepiej jest uzywac kopii niz wartosci wyjsciowej!!!"""

        plansza = plansza[:]

        # najlepsze pozycje wg mnie :) - komputer bedzie wybieral w kolejnosci najlepsze pozycje w ktorych zrobi ruch
        gdy_komputer_X = [(4, 2, 5), (4, 6, 8), (4, 8, 2), (4, 6, 8), (4, 6, 0), (4, 0, 6), (4, 6, 8), (4, 8, 6),
                            (4, 8, 2), (4, 2, 8), (4, 0, 6), (4, 6, 0)]
        gdy_komputer_O = (0, 2, 6, 8)

        #Sprawdzam cy komputer moze wygrac
        print("wybieram pole numer")

        for dobry_ruch in self.__dozwolony_ruch(plansza):
            plansza[dobry_ruch] = komputer
            if self.__jak_wygrac(plansza) == komputer:
                print(dobry_ruch)
                return dobry_ruch
            ##wycofanie tego ruchy aby zresetować plansze
            plansza[dobry_ruch] = self.__puste_pole
        # sprawdzamy czy w następnym ruchu może wygrać gracz. Komputer musi koniecznie zablokować ten ruch.
        # Kod w pętli dobiera kolejne elementy listy prawidłowych ruchów umieszczajac ruch czlowieka w kazdym po koleji pustym polu i sprawdza jego mozliwosc zwyciestwa
        # Jezeli sie okarze ruch da czlowiekowi zwyciestwo, komputer wykona ten ruch jako pierwszy i zablokuje pole na ruch czlowieka.
        # Jezeli nie bedzie takiej mozlwiosci, komputer wycofa ruch i sprawdzi kolejny najlepszy ruch z listy czy jest mozliwy do wykonania


        for dobry_ruch2 in self.__dozwolony_ruch(plansza):
            plansza[dobry_ruch2] = czlowiek
            if self.__jak_wygrac(plansza) == czlowiek:
                print(dobry_ruch2)
                return dobry_ruch2
            # ruch został sprawdzony
            plansza[dobry_ruch2] = self.__puste_pole

        ##jezeli nikt nie moze wygrac to komputer bedzie sprawdzal gdzie mozne postawic kolejny ruch o ile sa jeszcze wolne pola

        if komputer == "X":
            for i in range(len(gdy_komputer_X)):
                for ruch_x1 in gdy_komputer_X[i]:
                    if ruch_x1 in self.__dozwolony_ruch(plansza):
                        print(ruch_x1)
                        return ruch_x1
        elif komputer=="O":
            for ruch_x2 in gdy_komputer_O:
                if ruch_x2 in self.__dozwolony_ruch(plansza):
                    print(ruch_x2)
                    return ruch_x2
        print("trrter")

    def __koljeka(self, kolejka):
        """Zmien wykonawce ruchu"""
        if kolejka == "X":
            return "O"
        else:
            return "X"

    def __gratulacje(self, zwyciesca, komputer, czlowiek):
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

    def gra(self):
        self.instrukcja()
        self.czlowiek, self.komputer = self.__kto_kolko_a_kto_krzyzyk()
        self.kolejka = "X"
        self.plansza = self.__nowa_plansza()
        self.__wyswietl_plansze(self.plansza)
        print(f"Czlowiek: {self.czlowiek}, komputer: {self.komputer}")
        print(f"test: {self.__jak_wygrac(self.plansza)}")

        while not self.__jak_wygrac(self.plansza):
            if self.kolejka == self.czlowiek:
                self.ruch3 = self.__ruch_czlowieka(self.plansza)
                self.plansza[self.ruch3] = self.czlowiek
            else:
                self.ruch4 = self.__ruch_komputera(self.plansza, self.komputer, self.czlowiek)
                print(f"ruch4: {self.ruch4}")
                self.plansza[self.ruch4] = self.komputer
            self.__wyswietl_plansze(self.plansza)
            self.kolejka = self.__koljeka(self.kolejka)

        self.win = self.__jak_wygrac(self.plansza)
        self.__gratulacje(self.win, self.komputer, self.czlowiek)
        #self.zapisywanie = self.win
        self.text_file.writelines(self.win)
        self.text_file.close()

x = tictactoe()
x.gra()






















