# KNN Drift Detection

1.  **Zustandsmanagement (`get_state`, `set_state`):**
**Konfiguration (`config_scenario2.json`):**

```json
"drift_detectors": [
  {
    "type": "KNNDistanceMonitor",
    "delta": 10, // Wird überschrieben, wenn use_adaptive_thresholds=true und Stats geladen
    "lambda_threshold": 25, // Wird überschrieben, wenn use_adaptive_thresholds=true und Stats geladen
    "exit_threshold_factor": 0.7,
    "high_confidence_threshold": 0.9,
    "stable_known_classes": ["lego", "stone", "leaf"],
    "reference_stats_path": "./initial_state/knn_reference_stats.json", // SEHR WICHTIG
    "warm_up_samples": 5, // Niedrig, da wir Stats laden wollen
    "reference_update_interval": 100, // Parameter für die laufende Anpassung
    "reference_update_factor": 0.05,  // Parameter für die laufende Anpassung
    "pause_reference_update_during_drift": true,
    "drift_cooldown_period": 30,
    "use_adaptive_thresholds": false, // Standardmäßig AUS
    "allow_runtime_threshold_adaptation": false, // Standardmäßig AUS
    "adaptive_delta_std_multiplier": 1.5, // Faktor für adaptives Delta
    "adaptive_lambda_std_multiplier": 5.0 // Faktor für adaptives Lambda
  }
]
```

*   **Flexibilität:** Die neuen Flags `use_adaptive_thresholds` und `allow_runtime_threshold_adaptation` geben Kontrolle über das Verhalten des Detektors.
*   **`warm_up_samples: 5`:** Dieser Wert ist jetzt in primär auf das Laden der Statistiken über `reference_stats_path`. Wenn die Datei nicht gefunden wird oder fehlerhaft ist, würden diese 5 Samples für ein (sehr kurzes und unzuverlässiges) On-Device-Warm-up verwendet.
*   **Empfohlene Teststrategie mit der neuen Config:**
    1.  **Szenario A (Stabile, vorab definierte Schwellen, aber datengetriebener Start):**
        *   Setze `use_adaptive_thresholds: false`.
        *   Setze `allow_runtime_threshold_adaptation: false`.
        *   Stelle sicher, dass `reference_stats_path` auf eine valide Datei zeigt, die im `create_initial_knn_sc2.py` erstellt wurde. Der `KNNDistanceMonitor` wird `reference_mean` und `reference_std` aus dieser Datei laden.
        *   **Passe `delta` und `lambda_threshold` in der Config manuell an**, basierend auf den `suggested_delta` und `suggested_lambda`, die dein `create_initial_knn_sc2.py` Skript ausgibt (welche auf `reference_std` basieren).
        *   *Verhalten:* Der Monitor startet mit einem guten `reference_mean` und du hast `delta`/`lambda` basierend auf der initialen Datenverteilung gewählt. Die Schwellen bleiben während des Betriebs konstant. `reference_mean` und `reference_std` werden aber weiterhin langsam adaptiert (gesteuert durch `reference_update_interval/factor`).
    2.  **Szenario B (Adaptive Schwellen bei Initialisierung):**
        *   Setze `use_adaptive_thresholds: true`.
        *   Setze `allow_runtime_threshold_adaptation: false`.
        *   `reference_stats_path` muss auf eine valide Datei zeigen.
        *   Die `delta` und `lambda_threshold`-Werte in der Config werden ignoriert. Stattdessen werden sie bei der Initialisierung des Monitors aus der geladenen `reference_std` und den `adaptive_..._multiplier`-Werten berechnet.
        *   *Verhalten:* Der Monitor startet mit datengetriebenen Schwellen. Diese Schwellen bleiben dann während des Betriebs konstant. `reference_mean` und `reference_std` werden weiterhin langsam adaptiert.
    3.  **Szenario C (Volladaptive Schwellen):**
        *   Setze `use_adaptive_thresholds: true`.
        *   Setze `allow_runtime_threshold_adaptation: true`.
        *   `reference_stats_path` muss auf eine valide Datei zeigen.
        *   *Verhalten:* Der Monitor startet mit datengetriebenen Schwellen. Sowohl `reference_mean`, `reference_std` als auch `delta` und `lambda_threshold` werden im Laufe der Zeit basierend auf den neuen Daten und den Update-Intervallen/-Faktoren angepasst. Dies ist am flexibelsten, aber auch am schwierigsten zu kontrollieren, da sich die Sensitivität des Detektors ändern kann.