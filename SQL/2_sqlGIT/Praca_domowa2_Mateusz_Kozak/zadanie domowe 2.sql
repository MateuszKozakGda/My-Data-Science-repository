SELECT COUNT(*)
FROM wnioski
WHERE kwota_rekompensaty > 400
AND kwota_rekompensaty < 500;

SELECT COUNT(o.id_wniosku)*100.0 /
       (SELECT COUNT(*) FROM wnioski) as procent_wnioskow_z_analiza
FROM wnioski w, analiza_operatora o
WHERE w.id = o.id_wniosku;


SELECT MAX(kwota_rekompensaty)
FROM wnioski w, analiza_prawna a
WHERE w.id = a.id_wniosku
AND status = 'zaakceptowany'
;