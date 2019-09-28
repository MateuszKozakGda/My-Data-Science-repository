--- a) ile wnioskow pochodzi z niezakloconej podrozy
with ilosc_zaklocen as(
SELECT w.id, aw.zaklocenie_ok
FROM wnioski w
LEFT JOIN podroze p  on w.id = p.id_wniosku
LEFT JOIN analizy_wnioskow aw on w.id = aw.id_wniosku
WHERE aw.zaklocenie_ok = TRUE)

SELECT COUNT(1) as ilosc_wnioskow_z_zakloceniem
FROM ilosc_zaklocen
;

--- b) W ktrym roku byo najwiecej wnioskw w jezyku polskim;

with funkcja as(
SELECT COUNT(1) as liczba_wnioskow_w_jpolskim, extract(year from data_utworzenia)::numeric as rok, jezyk
FROM wnioski
WHERE jezyk LIKE 'pl'
GROUP BY extract(year from data_utworzenia)::numeric, jezyk
ORDER BY extract(year from data_utworzenia)::numeric)

SELECT rok, MAX(liczba_wnioskow_w_jpolskim)
FROM funkcja
WHERE liczba_wnioskow_w_jpolskim = (SELECT MAX(liczba_wnioskow_w_jpolskim) FROM funkcja)
GROUP BY rok;


--- c) ktore wnioski podczas analizy nie wykazaly obecnosci strajku oraz dobrej pogody?:
SELECT w.id, aw.brak_strajku, aw.pogoda_ok
FROM wnioski w
LEFT JOIN analizy_wnioskow aw on w.id = aw.id_wniosku
WHERE aw.brak_strajku IS TRUE
AND aw.pogoda_ok IS FALSE;

--- d) Wyswietl dane o zmanie ilosci analiz prawnych z roku na rok
SELECT COUNT(id) as ilosc_analiz, extract(YEAR from data_rozpoczecia)::numeric as rok,
       LAG(COUNT(1)) over () as liczba_wnioskow,
       COUNT(1) - LAG(COUNT(1)) over () as zmiana_ilosci_wnioskow
FROM analiza_prawna
GROUP BY extract(YEAR from data_rozpoczecia)::numeric
ORDER BY extract(YEAR from data_rozpoczecia)::numeric;

--- e) Podaj mediane i kwartyle Q1 i Q3 procentu oplaty na ubsluge wnioskow z analiza prawna

with percentyle as(

  with nowa_funkcja as(
SELECT ap.koszt as koszt_postepowania
FROM wnioski w, analiza_prawna ap
WHERE w.id = ap.id_wniosku)

SELECT koszt_postepowania * 100.0 / (SELECT SUM(koszt_postepowania) FROM nowa_funkcja) as procent_kosztow
FROM nowa_funkcja)

SELECT percentile_disc(0.25) within group ( order by procent_kosztow ) as kwartyl_Q1,
       percentile_disc(0.75) within group ( order by procent_kosztow ) as kwartyl_03,
       percentile_disc(0.5) within group ( order by procent_kosztow ) as mediana
FROM percentyle
;

--- f) Jaka jest korelacja miedzy kwota rekompensaty oryginalnej i kwota rekompensaty dla jezyka?
SELECT CORR(kwota_rekompensaty_oryginalna, kwota_rekompensaty) as korelcaja_kwot, jezyk
FROM wnioski
GROUP BY jezyk;
