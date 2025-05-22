# Autonomous On-Device Adaptation - Simplified Sequence Diagram

This document provides a simplified UML sequence diagram for scientific publications, focusing on the core concepts of autonomous on-device adaptation in TinyLCM.

## Simplified Sequence Diagram

```plantuml
@startuml Autonomous On-Device Adaptation Sequence
!theme plain
skinparam backgroundColor white
skinparam sequenceParticipantBackgroundColor white
skinparam sequenceParticipantBorderColor black
skinparam sequenceArrowColor black

participant "Pipeline" as Pipeline
participant "DriftDetector" as Monitor  
participant "QuarantineBuffer" as Buffer
participant "HeuristicAdapter" as Adapter
participant "Classifier" as Classifier

== Drift Detection ==
Pipeline -> Monitor: update(sample)
Monitor -> Monitor: Page-Hinkley Test
Monitor -> Pipeline: drift_detected

== Quarantine ==
Pipeline -> Buffer: add_sample(features, prediction)

== Heuristic Analysis ==
note over Pipeline: Periodic Analysis
Pipeline -> Adapter: analyze_quarantine()
Adapter -> Buffer: get_samples()
Adapter -> Adapter: cluster_samples()
Adapter -> Adapter: generate_pseudo_labels()
Adapter -> Pipeline: pseudo_labeled_samples

== Adaptation ==
Pipeline -> Classifier: add_training_samples(pseudo_labels)
Pipeline -> Classifier: retrain()

alt performance_check
    Pipeline -> Pipeline: rollback()
else
    Pipeline -> Pipeline: adaptation_success
end

@enduml
```

## Key Phases

1. **Drift Detection**: Page-Hinkley test detects distribution changes in neighbor distances
2. **Quarantine**: Suspicious samples are temporarily stored for analysis
3. **Heuristic Analysis**: Clustering-based pseudo-labeling of quarantined samples
4. **Adaptation**: Cautious classifier update with rollback capability

This simplified view focuses on the essential autonomous adaptation cycle suitable for scientific documentation.