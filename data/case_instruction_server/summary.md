# Case Instruction Server

- target_total_cases: 5000
- generation_target: 5000
- seed_cases: 161
- issued: 3803
- submitted: 3760
- training_cases_current: 3760
- remaining: 1240

## Coverage

### persona
- enfant: current=684 target=900.0 gap=216.0
- conjoint: current=456 target=600.0 gap=144.0
- beau_enfant: current=342 target=450.0 gap=108.0
- fratrie: current=304 target=400.0 gap=96.0
- notaire: current=304 target=400.0 gap=96.0
- avocat: current=267 target=350.0 gap=83.0
- partenaire_pacs: current=267 target=350.0 gap=83.0
- concubin: current=228 target=300.0 gap=72.0
- associe: current=267 target=350.0 gap=83.0
- petit_enfant: current=190 target=250.0 gap=60.0
- tiers: current=190 target=250.0 gap=60.0
- narrateur_neutre: current=304 target=400.0 gap=96.0

### voice
- premiere_personne: current=1711 target=2250.0 gap=539.0
- troisieme_personne: current=1330 target=1750.0 gap=420.0
- note_dossier: current=381 target=500.0 gap=119.0
- parole_rapportee: current=381 target=500.0 gap=119.0

### format
- question_directe: current=836 target=1100.0 gap=264.0
- mail_brouillon: current=684 target=900.0 gap=216.0
- recit_libre: current=836 target=1100.0 gap=264.0
- note_professionnelle: current=533 target=700.0 gap=167.0
- oral_retranscrit: current=533 target=700.0 gap=167.0
- message_conflictuel: current=381 target=500.0 gap=119.0

### length_band
- court: current=685 target=900.0 gap=215.0
- moyen: current=1596 target=2100.0 gap=504.0
- long: current=1217 target=1600.0 gap=383.0
- tres_long: current=305 target=400.0 gap=95.0

### noise
- propre: current=1596 target=2100.0 gap=504.0
- legeres_fautes: current=836 target=1100.0 gap=264.0
- fautes_et_abreviations: current=647 target=850.0 gap=203.0
- ambigu: current=609 target=800.0 gap=191.0
- tres_brouillon: current=115 target=150.0 gap=35.0

### numeric_density
- sans_montant: current=229 target=300.0 gap=71.0
- un_montant: current=989 target=1300.0 gap=311.0
- plusieurs_montants: current=1444 target=1900.0 gap=456.0
- montants_et_dates: current=1141 target=1500.0 gap=359.0

### date_precision
- aucune: current=571 target=750.0 gap=179.0
- approx: current=761 target=1000.0 gap=239.0
- exacte: current=2471 target=3250.0 gap=779.0

### complexity
- simple: current=761 target=1000.0 gap=239.0
- intermediaire: current=1520 target=2000.0 gap=480.0
- complexe: current=913 target=1200.0 gap=287.0
- hard_negative: current=609 target=800.0 gap=191.0

### primary_topic
- ordre_heritiers: current=304 target=400.0 gap=96.0
- famille_recomposee: current=456 target=600.0 gap=144.0
- regimes_matrimoniaux: current=304 target=400.0 gap=96.0
- donations_reduction: current=380 target=500.0 gap=120.0
- assurance_vie: current=380 target=500.0 gap=120.0
- indivision_partage: current=342 target=450.0 gap=108.0
- entreprise_dutreil: current=304 target=400.0 gap=96.0
- demembrement_usufruit: current=229 target=300.0 gap=71.0
- testament_legs: current=304 target=400.0 gap=96.0
- dettes_passif: current=229 target=300.0 gap=71.0
- pacs_concubinage: current=267 target=350.0 gap=83.0
- international_procedure: current=304 target=400.0 gap=96.0

### hard_negative_mode
- pas_de_deces_clair: current=183 target=240.0 gap=57.0
- infos_incompletes: current=182 target=240.0 gap=58.0
- faits_contradictoires: current=152 target=200.0 gap=48.0
- hors_perimetre_mal_qualifie: current=92 target=120.0 gap=28.0

### hard_negative_intensity
- soft: current=487 target=640.0 gap=153.0
- hard: current=122 target=160.0 gap=38.0
