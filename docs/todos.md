# OPL List of Todos

- TODO: Testen von der Driftzurücksetzung des KNNDistanceMonitor

- TODO: Fix Connectivity Trend Chart auf der Homepage. Die active Devices werden nicht mehr richtig angezeigt. -> Methode wie man die richtig bekommt ist in Der device Liste 

- TODO: Device Seite überarbieten. Avg Inference Time, Avg CPU Usage und Avg Memory Usage Status Cards sind nicht nützlich, wenn wir mehre geräte haben. Daher ein Chart erstellen, wo pro device diese metriken im zeitverklauf abgebildet werden. Siehe dazu anlog wie Model Performance chart aufgesetzt wurde

- TODO: Device Liste: Package Icon in Actions entfernen

- TODO: Aus dem Model Performance Chart die System Metriken entfernen, diese sind Device spezifisch und nicht Model spezifisch

- TODO: Data Hub Seite fixen, es werden wenn man auf die seite geht und all devices angewählt hat die daten nicht richtig geladen. Wenn man dann device auswählt, ist ein prediction typoe automatisch drin, dieser zeigt aber nicht die richtigen bilder an, erst, nachdem man den prediction type gewechselt hat, lädt dies richtig. -> Fixen, dass die Daten immer richtig geladen werden

- TODO: Start und End Date filter auf Data Hub (alle Tabs) geht nicht

- TODO: Möglichkeit zum Löschen von Daten im Data Hub, mit Multi Select Option. -> Daten sollen dann auch aus MINIO rausgelöscht werden

- TODO: Models Performance Chart auf der Models Page geht nicht richtig, baiserd imme rnur auf dem letzten run -> analog machen wie das auf Homepage

- TODO: Model Registry, passt das rendering des Device Namen nicht, es zeigt immer pi an, die id schneidet es aber ab?

- TODO: Zu viele Notifictaions. Keine Notification, wenn ein package erfolgreich verarbeitet wurde. -> nur wenn ein Fehler auftritt

- TODO: Autorefresh auf der Drift Page entfernen

- TODO: Drift page neu aufbauen. So wie die Anderen sietn auch, mit mehren einzelen charts.

- TODO:  Drift logging der metriken überdenken -> Besser nur ein Run pro Device, und alles darunter?

- TODO: Model Packages logik prüfen, die Hasprüfung geht nicht richtig, neue Moodelle (-Versionen) werden nicht erkannt, und dann gesendet und in die registry genommen.

- TODO: Loggen von allen Modells -> TFLITE, PCA und KNN --> Logik überlegen, wie man dann das gtracking sinnvoll macht. Einzel meztriken ziehen, oder kommuliert für alle und per device ausgeben?

- TODO: Admin Page bauen, zb mit Möglichkeit, Devices zu registrieren. Dort sollte dann ein API Key generiert werden, denn man dann dem device geben kann. nur mit diesem key, kann sich das Gerät auf dem Server registrieren. -> API Key generieren und in die DB speichern, dann kann sich das Gerät registrieren. Dann aber Device ID Überdenken, macht dann keinen sinn mehr, da die on deivce erstellt wird. --> Besser, name des device nutzen als Device name. Dieser wird in der Admin Fläche angegeben und hat dann den key für auth.

- TODO: Die Drifgt Samples Daten in die Drift Events lIste mappen, so dass ich mir die Bilder zu dem Event dort anscheun kann. mit Option, diese direkt runterzuladen.

- TODO: Model deployment prüfen. TF Lite und PCA kann offline trainiert werden und in der registry geloggt werden. wenn tag prod, dann kann das modell auf das device geschoben werden --> Pull, da device sich connected wenn es will, und nicht on device ein webserver gemacht werden sll zum energy sparen.

- TODO: Retraining Pipeline prüfen: Wenn ich daten im data hub habe, dann will ich diese nehmen können, und sage, dass ist mein neues set an Daten, die ich für ein Retraining will. Dann soll das eine pipeline triggern, die die modelle automatisch trainiert, und evaluiert und dann in die registry schibt, und für depolyoment tagt.

- TODO: User Management erstellen. 


- TODO: App Config (Thresholds,Drift Params, solllten au9f Platfrom ersichtlich sein und auch pro device / Modell anpassbar sein) --> Dann wieder pull und update on device

- TODO: Self Customization der Homepage. Der USer soll sein Dahboard slebt anpassen können. Dazu dieses in Sections und Spalten unterteilen. Dann kann in diese Charts / Tabellen die zur Verfügung sind, "hineingezogen" werden. Diese werden dann in der DB gespeichert und beim laden der Seite wieder angezeigt (User Management vorrausgesetzt für individuelle, sonst oer deoplyoment).

- TODO: Im Data Hub die Logs müssen aggregiert werdden. Device sendet je nach einstellung alle paar sekunden ioder mInuten die Logs. Wir hjaben immer eine Session ID, für diese dann aber zb. 100 Logs. Es soll für eine Session ID imme rnur ein gesamtes log geben. Dieses muss daher im BAckend immer aufgemacht werden, und die neuen logs hinzugefügt werdne. -> Ebenfalls option, logs zu löschen, dann ebenfalls aus minio löschen. 

- TODO: Spinx dokumentation erstellen und in frontend einbinden. Sowie Quickstart Examples guide erstellen.

- TODO: Data versioning muss implementiert werden. Data Hub auch um Training Data ergänzen, soll als mein warehouse dienen. Plus dort kann ich Daten von prediction oder drift in den training data hub verschieben. -> Sieh auch TODO zu retraining pipeline, diese baut dann hierauf auf.

- 