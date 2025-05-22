# Autonomous On-Device Adaptation - Sequence Diagram

This document provides a UML sequence diagram illustrating the complete sequence of operations during an autonomous on-device adaptation cycle in TinyLCM, from drift detection to classifier update.

## Overview

The sequence diagram shows how the system autonomously detects drift, quarantines suspicious samples, applies heuristic pseudo-labeling, and cautiously updates the classifier while maintaining rollback capabilities. This process occurs entirely on-device without requiring external supervision or ground truth labels.

## Sequence Diagram

```plantuml
@startuml Autonomous On-Device Adaptation Sequence
!theme plain
skinparam backgroundColor white
skinparam sequenceParticipantBackgroundColor white
skinparam sequenceParticipantBorderColor black
skinparam sequenceArrowColor black

participant "AdaptivePipeline" as Pipeline
participant "KNNDistanceMonitor" as Monitor  
participant "QuarantineBuffer" as Buffer
participant "HeuristicAdapter" as Adapter
participant "LightweightKNN" as Classifier
participant "StateManager" as State
participant "AdaptationTracker" as Tracker

== Drift Detection Phase ==
Pipeline -> Pipeline: process(image)
Pipeline -> Monitor: update(record)
Monitor -> Monitor: _calculate_ph_statistics()
Monitor -> Monitor: drift_detected = True
Monitor -> Pipeline: _notify_callbacks(drift_info)

== Quarantine Phase ==
Pipeline -> Buffer: should_quarantine(drift_info)
Buffer -> Pipeline: True
Pipeline -> Buffer: add_sample(features, prediction, drift_info)
Buffer -> Buffer: buffer.append(entry)
Pipeline -> Tracker: log_quarantine(sample_id, reason)

== Periodic Heuristic Analysis ==
note over Pipeline: samples_since_quarantine_check >= interval
Pipeline -> Pipeline: run_heuristic_adaptation_cycle()
Pipeline -> State: create_snapshot("before_adaptation")
State -> Classifier: get_state()
State -> State: _save_snapshot(states)

== Pseudo-Label Generation ==
Pipeline -> Adapter: apply_pseudo_labels(min_confidence)
Adapter -> Buffer: get_data()
Buffer -> Adapter: quarantined_samples[]
Adapter -> Adapter: _group_by_prediction(samples)
loop for each prediction_group
    Adapter -> Adapter: _calculate_dispersion(group)
    alt dispersion <= variance_threshold
        Adapter -> Adapter: _generate_potential_label()
        Adapter -> Adapter: _select_representatives()
    end
end
Adapter -> Buffer: mark_as_processed(sample_ids)
Adapter -> Pipeline: adaptation_samples[]

== Classifier Update ==
loop for each adaptation_sample
    Pipeline -> Classifier: add_training_sample(features, pseudo_label)
    Pipeline -> Classifier: retrain()
    Pipeline -> Tracker: log_adaptation_event(sample, old_label, new_label)
end

== Performance Check & Rollback ==
Pipeline -> Adapter: rollback_needed()
alt performance_degraded
    Adapter -> Pipeline: True
    Pipeline -> State: load_snapshot(snapshot_id)
    State -> Classifier: set_state(saved_state)
    Pipeline -> Tracker: log_rollback(reason)
else
    Adapter -> Pipeline: False
    Pipeline -> Tracker: log_successful_adaptation()
end

@enduml
```

## Key Phases Explained

### 1. Drift Detection Phase
- **Page-Hinkley Test**: The KNNDistanceMonitor continuously monitors neighbor distances using the Page-Hinkley algorithm
- **Callback Mechanism**: When drift is detected, callbacks notify the pipeline to initiate quarantine procedures

### 2. Quarantine Phase  
- **Sample Storage**: Suspicious samples are stored in the QuarantineBuffer with metadata (features, prediction, timestamp)
- **Tracking**: All quarantine events are logged for analysis and debugging

### 3. Periodic Heuristic Analysis
- **Trigger Condition**: Analysis occurs when `samples_since_quarantine_check >= quarantine_check_interval`
- **State Snapshot**: System state is preserved before making any adaptations to enable rollback

### 4. Pseudo-Label Generation
- **Clustering Analysis**: Quarantined samples are grouped by prediction and analyzed for coherence
- **Dispersion Check**: Groups with low intra-cluster dispersion are considered for pseudo-labeling
- **Representative Selection**: The most representative samples from coherent clusters are selected for adaptation

### 5. Classifier Update
- **Cautious Addition**: Only high-confidence pseudo-labeled samples are added to the training set
- **Incremental Retraining**: The classifier is retrained with the new samples
- **Event Logging**: All adaptation events are tracked for auditability

### 6. Performance Check & Rollback
- **Performance Monitoring**: The system evaluates whether the adaptation improved or degraded performance
- **Automatic Rollback**: If performance degrades, the system automatically reverts to the previous state
- **Adaptation Tracking**: Success or failure of adaptations is logged for future analysis

This autonomous cycle enables edge devices to adapt to new conditions while maintaining robustness through cautious heuristics and rollback mechanisms.