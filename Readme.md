## Opis Projektu ##
Celem projektu jest utworzenie zbioru danych gestów alfabetu migowego przy użyciu [narzędzia](https://github.com/kamilmlodzikowski/WZUM_2023_DataGatherer), a następnie przygotowanie modelu rozpoznającego gesty.

### Zebranie danych ###
Przy wykorzystaniu wspomnianego narzędzia oraz współpracy wszystkich studentów udało się utworzyć zbiór danych zawierający ponad 5000 rekordów.

### Opis rozwiązania ###

Rozwiązanie problemu wykorzystało model **LogisticRegression**, który wykazał się wysoką dokładnością w testach. W ramach preprocessingu usunięte zostały kolumny o nazwie ***'multi_hand_world_landmarks'***, które nie wniosły znaczących informacji w porównaniu do danych pochodzących z ***'multi_hand_landmarks'***. Zastosowano **LabelEncoding** w celu zamiany kolumny ***'handedness.label'*** oraz ***'letter'*** na wartości liczbowe.

Do redukcji liczby cech użyto **SelectKBest**, który wybrał tylko te cechy, które niosą ze sobą najwięcej informacji (48). Dodatkowo, wartości zostały przeskalowane za pomocą **StandardScaler**.